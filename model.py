"""
GraphDTA model architecture: GIN (drug) + CNN (protein) + cross-attention.
Extracted from the training notebook — do not modify without retraining,
since checkpoints are tied to this exact class structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GINDrugEncoder(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=32, output_dim=128, num_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        return F.relu(self.fc(x))


class CNNProteinEncoder(nn.Module):
    """Returns per-residue features (no pooling) so attention can select over them."""
    def __init__(self, vocab_size=26, embed_dim=128, num_filters=32, kernel_sizes=(4, 6, 8)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding='same') for k in kernel_sizes
        ])
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        conv_outs = [F.relu(c(emb)) for c in self.convs]
        residue_features = torch.cat(conv_outs, dim=1)
        return residue_features.permute(0, 2, 1)   # [batch, seq_len, out_dim]


class DrugProteinAttention(nn.Module):
    """Cross-attention: drug vector (query) attends over per-residue protein features (key/value)."""
    def __init__(self, drug_dim, protein_dim, attn_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(drug_dim, attn_dim)
        self.key_proj = nn.Linear(protein_dim, attn_dim)
        self.value_proj = nn.Linear(protein_dim, protein_dim)

    def forward(self, drug_vec, residue_features, protein_seq):
        query = self.query_proj(drug_vec).unsqueeze(1)
        keys = self.key_proj(residue_features)
        values = self.value_proj(residue_features)

        scores = torch.bmm(query, keys.transpose(1, 2)) / (keys.size(-1) ** 0.5)
        mask = (protein_seq == 0).unsqueeze(1)
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights, values).squeeze(1)
        return pooled, attn_weights.squeeze(1)


class GraphDTA(nn.Module):
    def __init__(self, drug_output_dim=128, protein_output_dim=128):
        super().__init__()
        self.drug_encoder = GINDrugEncoder(output_dim=drug_output_dim)
        self.protein_encoder = CNNProteinEncoder()
        self.attention = DrugProteinAttention(
            drug_dim=drug_output_dim, protein_dim=self.protein_encoder.out_dim
        )
        self.protein_fc = nn.Linear(self.protein_encoder.out_dim, protein_output_dim)

        combined_dim = drug_output_dim + protein_output_dim
        self.fc1 = nn.Linear(combined_dim, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(64, 1)

    def forward(self, data, return_attention=False):
        drug_vec = self.drug_encoder(data.x, data.edge_index, data.batch)
        residue_features = self.protein_encoder(data.target)
        protein_vec, attn_weights = self.attention(drug_vec, residue_features, data.target)
        protein_vec = F.relu(self.protein_fc(protein_vec))

        x = F.relu(self.fc1(torch.cat([drug_vec, protein_vec], dim=1)))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        prediction = self.out(x).squeeze(-1)

        if return_attention:
            return prediction, attn_weights
        return prediction

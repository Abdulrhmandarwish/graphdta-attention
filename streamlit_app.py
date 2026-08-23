"""
GraphDTA Streamlit app — predicts drug-target binding affinity and visualizes
which protein residues the model attended to most.

Checkpoints are pulled from Hugging Face Hub at startup:
https://huggingface.co/Abdulrhman2002/graphdta-checkpoints
"""

import streamlit as st
import torch
import numpy as np
from huggingface_hub import hf_hub_download

from model import GraphDTA
from data_utils import smiles_to_graph, encode_protein, MAX_SEQ_LEN

REPO_ID = "Abdulrhman2002/graphdta-checkpoints"
device = torch.device("cpu")

_scale_notes = {
    "Davis (pKd)": "Higher = stronger binding. Typical range ~5 to ~10.8.",
    "KIBA (KIBA score)": "Composite score (Ki/Kd/IC50 combined). Higher = stronger binding.",
}

@st.cache_resource
def load_models():
    models = {}
    for dataset_key, filename in [("Davis (pKd)", "best_model_davis.pt"), ("KIBA (KIBA score)", "best_model_kiba.pt")]:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        model = GraphDTA().to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[dataset_key] = model
    return models

models = load_models()

st.title("GraphDTA: Drug–Target Binding Affinity Predictor")
st.markdown(
    "GIN-based graph neural network (drug) + CNN with cross-attention (protein). "
    "Enter a drug SMILES string and a protein sequence to predict binding affinity "
    "and see which residues the model focused on.\n\n"
    f"[Model weights on Hugging Face]({'https://huggingface.co/' + REPO_ID})"
)

# Example Data
EXAMPLE_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
EXAMPLE_PROTEIN = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLI"

if st.button("Load Example (Aspirin + AAK1)"):
    st.session_state.smiles_input = EXAMPLE_SMILES
    st.session_state.protein_input = EXAMPLE_PROTEIN
    st.session_state.dataset_choice = "Davis (pKd)"
    st.session_state.top_k = 10

smiles = st.text_input(
    "Drug SMILES", 
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)",
    key="smiles_input"
)
protein_sequence = st.text_area(
    "Protein sequence (amino acids)", 
    placeholder="e.g. MKKFFDSRREQGGSGLGSG...",
    height=150,
    key="protein_input"
)
dataset_choice = st.radio(
    "Model / training dataset", 
    ["Davis (pKd)", "KIBA (KIBA score)"],
    key="dataset_choice"
)
top_k = st.slider(
    "Top-K residues to highlight", 
    min_value=3, max_value=25, value=10, step=1,
    key="top_k"
)

if st.button("Predict", type="primary"):
    if not smiles.strip() or not protein_sequence.strip():
        st.error("Please provide both a SMILES string and a protein sequence.")
    else:
        try:
            graph = smiles_to_graph(smiles.strip())
            protein_encoded = encode_protein(protein_sequence.strip())

            # Build a single-sample PyG batch manually
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            graph.target = torch.tensor([protein_encoded], dtype=torch.long)
            graph = graph.to(device)

            model = models[dataset_choice]
            with torch.no_grad():
                prediction, attn_weights = model(graph, return_attention=True)

            pred_value = prediction.item()
            weights = attn_weights[0].cpu().numpy()

            # Only consider real (non-padding) residue positions
            real_len = min(len(protein_sequence.strip()), MAX_SEQ_LEN)
            weights = weights[:real_len]

            top_k_int = int(top_k)
            top_positions = weights.argsort()[::-1][:top_k_int]
            top_residues = [
                f"Position {p+1}: '{protein_sequence[p]}' (weight={weights[p]:.4f})"
                for p in sorted(top_positions)
            ]

            # Display prediction
            st.markdown(f"**Predicted affinity ({dataset_choice}):** {pred_value:.3f}")
            st.markdown(f"_{_scale_notes[dataset_choice]}_")
            
            st.markdown(f"**Top {top_k_int} attended residues:**")
            st.markdown("\n".join(["- " + r for r in top_residues]))

            # Highlighted sequence for visual inspection
            highlight_set = set(top_positions.tolist())
            highlighted = "".join(
                f"**[{c}]**" if i in highlight_set else c
                for i, c in enumerate(protein_sequence[:real_len])
            )
            
            st.markdown("**Highlighted sequence (attended residues):**")
            st.markdown(highlighted)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

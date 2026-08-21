"""
GraphDTA Gradio app — predicts drug-target binding affinity and visualizes
which protein residues the model attended to most.

Checkpoints are pulled from Hugging Face Hub at startup:
https://huggingface.co/Abdulrhman2002/graphdta-checkpoints
"""

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from model import GraphDTA
from data_utils import smiles_to_graph, encode_protein, MAX_SEQ_LEN

REPO_ID = "Abdulrhman2002/graphdta-checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load both trained models once at startup ----
_models = {}
_scale_notes = {
    "Davis (pKd)": "Higher = stronger binding. Typical range ~5 to ~10.8.",
    "KIBA (KIBA score)": "Composite score (Ki/Kd/IC50 combined). Higher = stronger binding.",
}

def load_model(dataset_key, filename):
    path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    model = GraphDTA().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _models[dataset_key] = model

load_model("Davis (pKd)", "best_model_davis.pt")
load_model("KIBA (KIBA score)", "best_model_kiba.pt")


def predict(smiles, protein_sequence, dataset_choice, top_k):
    if not smiles.strip() or not protein_sequence.strip():
        return "Please provide both a SMILES string and a protein sequence.", None

    try:
        graph = smiles_to_graph(smiles.strip())
    except Exception as e:
        return f"Couldn't parse SMILES: {e}", None

    protein_encoded = encode_protein(protein_sequence.strip())

    # Build a single-sample PyG batch manually
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph.target = torch.tensor([protein_encoded], dtype=torch.long)
    graph = graph.to(device)

    model = _models[dataset_choice]
    with torch.no_grad():
        prediction, attn_weights = model(graph, return_attention=True)

    pred_value = prediction.item()
    weights = attn_weights[0].cpu().numpy()

    # Only consider real (non-padding) residue positions
    real_len = min(len(protein_sequence.strip()), MAX_SEQ_LEN)
    weights = weights[:real_len]

    top_k = int(top_k)
    top_positions = weights.argsort()[::-1][:top_k]
    top_residues = [
        f"Position {p+1}: '{protein_sequence[p]}' (weight={weights[p]:.4f})"
        for p in sorted(top_positions)
    ]

    result_text = (
        f"**Predicted affinity ({dataset_choice}):** {pred_value:.3f}\n\n"
        f"_{_scale_notes[dataset_choice]}_\n\n"
        f"**Top {top_k} attended residues:**\n" + "\n".join(top_residues)
    )

    # Highlighted sequence for visual inspection
    highlight_set = set(top_positions.tolist())
    highlighted = "".join(
        f"**[{c}]**" if i in highlight_set else c
        for i, c in enumerate(protein_sequence[:real_len])
    )

    return result_text, highlighted


with gr.Blocks(title="GraphDTA — Binding Affinity Predictor") as demo:
    gr.Markdown(
        "# GraphDTA: Drug–Target Binding Affinity Predictor\n"
        "GIN-based graph neural network (drug) + CNN with cross-attention (protein). "
        "Enter a drug SMILES string and a protein sequence to predict binding affinity "
        "and see which residues the model focused on.\n\n"
        f"[Model weights on Hugging Face]({'https://huggingface.co/' + REPO_ID})"
    )

    with gr.Row():
        with gr.Column():
            smiles_input = gr.Textbox(
                label="Drug SMILES",
                placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)",
            )
            protein_input = gr.Textbox(
                label="Protein sequence (amino acids)",
                placeholder="e.g. MKKFFDSRREQGGSGLGSG...",
                lines=6,
            )
            dataset_choice = gr.Radio(
                choices=["Davis (pKd)", "KIBA (KIBA score)"],
                value="Davis (pKd)",
                label="Model / training dataset",
            )
            top_k = gr.Slider(minimum=3, maximum=25, value=10, step=1, label="Top-K residues to highlight")
            submit_btn = gr.Button("Predict", variant="primary")

        with gr.Column():
            output_text = gr.Markdown(label="Prediction")
            output_highlight = gr.Markdown(label="Highlighted sequence (attended residues)")

    submit_btn.click(
        predict,
        inputs=[smiles_input, protein_input, dataset_choice, top_k],
        outputs=[output_text, output_highlight],
    )

    gr.Examples(
        examples=[
            ["CC(=O)Oc1ccccc1C(=O)O",
             "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLI",
             "Davis (pKd)", 10],
        ],
        inputs=[smiles_input, protein_input, dataset_choice, top_k],
    )

if __name__ == "__main__":
    demo.launch()

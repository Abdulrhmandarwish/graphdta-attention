# GraphDTA — Drug-Target Binding Affinity Prediction (with Attention)

A PyTorch reproduction of [GraphDTA](https://github.com/thinng/GraphDTA) (Nguyen et al., 2020),
extended with a cross-attention layer between the drug and protein branches for
interpretability — surfacing which protein residues the model attends to for a
given drug.

## Architecture

- **Drug branch:** Graph Isomorphism Network (GIN) over the molecule graph
  (atoms as nodes, bonds as edges), built from SMILES via RDKit.
- **Protein branch:** 1D CNN over the amino acid sequence, kept at per-residue
  resolution (no pooling).
- **Cross-attention:** the drug representation queries the per-residue protein
  features, producing an attention weight per residue — used both to build the
  combined representation and to interpret which residues mattered most for a
  given prediction.
- **Head:** concatenation of drug + attended protein vectors → fully connected
  layers → single affinity prediction.

## Results

| Dataset | Model | CI | MSE |
|---|---|---|---|
| Davis | This repo (GIN + Attention) | **0.872** | **0.285** |
| Davis | GraphDTA paper (GIN) | 0.882 | 0.147 |
| Davis | GraphDTA paper (GAT_GCN) | 0.891 | 0.139 |
| Davis | DeepDTA (paper) | 0.878 | 0.261 |
| KIBA | This repo (GIN + Attention) | **0.834** | **0.222** |
| KIBA | GraphDTA paper (GIN) | 0.840 | 0.147 |

**Note:** CI (ranking ability) closely matches the original paper on both
datasets (within ~0.01). MSE is higher than the paper, likely due to the
added attention layer changing optimization dynamics, and no extensive
hyperparameter tuning being performed (the paper's numbers reflect tuned
learning rates/schedules not reproduced here).

## Live Demo

[Hugging Face Space](#) — _add the link once deployed_

Model checkpoints: [huggingface.co/Abdulrhman2002/graphdta-checkpoints](https://huggingface.co/Abdulrhman2002/graphdta-checkpoints)

## Running Locally

```bash
git clone <this-repo-url>
cd graphdta-project
pip install -r requirements.txt
python app.py
```

## Project Structure

```
graphdta-project/
├── app.py            # Gradio app
├── model.py           # Model architecture (GIN + CNN + attention)
├── data_utils.py       # SMILES -> graph, protein sequence encoding
├── requirements.txt
├── notebooks/          # Training notebook(s)
└── checkpoints/        # Local copies of trained weights (optional; see .gitignore)
```

## Credit

Original method: Nguyen, T. et al. "GraphDTA: predicting drug-target binding
affinity with graph neural networks." *Bioinformatics*, 2021.
Original repo: [github.com/thinng/GraphDTA](https://github.com/thinng/GraphDTA)
Datasets: Davis et al. (2011), KIBA (Tang et al., 2014).

"""
Preprocessing utilities: SMILES -> graph, protein sequence -> encoded array.
Extracted from the training notebook — must stay identical to what the
model was trained on.
"""

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
]

SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOCAB)}
MAX_SEQ_LEN = 1000


def one_hot_encoding(value, allowed_values):
    if value not in allowed_values:
        value = allowed_values[-1]
    return [1.0 if value == v else 0.0 for v in allowed_values]


def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetSymbol(), ATOM_LIST)
    features += one_hot_encoding(atom.GetDegree(), list(range(11)))
    features += one_hot_encoding(atom.GetTotalNumHs(), list(range(11)))
    features += one_hot_encoding(atom.GetValence(Chem.ValenceType.IMPLICIT), list(range(11)))
    features += [1.0 if atom.GetIsAromatic() else 0.0]
    return features


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit couldn't parse SMILES: {smiles}")

    atom_features = [get_atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j]); edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def encode_protein(sequence: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    encoding = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(sequence[:max_len]):
        encoding[i] = SEQ_DICT.get(ch, 0)
    return encoding

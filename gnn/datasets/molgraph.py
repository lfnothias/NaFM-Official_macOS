import pickle
import numpy as np
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import time

ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
FORMAL_CHARGE = [-1, -2, 1, 2, 0]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


class MolGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MolGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["clean_smiles.pkl"]

    @property
    def processed_file_names(self):
        return ["clean_mols.pt"]

    @staticmethod
    def get_scaffold(mol):
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if len(core.GetAtoms()) < 5:
            core = mol
        scaffold = mol.GetSubstructMatches(core)
        scaffold_mask = torch.zeros((mol.GetNumAtoms(),), dtype=torch.bool)
        for atom in scaffold[0]:
            scaffold_mask[atom] = True
        return scaffold_mask

    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            smiles_list = pickle.load(open(path, "rb"))
            samples = []

            for smiles in tqdm(smiles_list):
                mol = Chem.MolFromSmiles(smiles)

                atom_feats = []
                for atom in mol.GetAtoms():
                    atom_feats.append(
                        [
                            ATOM_LIST.index(atom.GetAtomicNum()),
                            CHIRALITY_LIST.index(atom.GetChiralTag())
                            if atom.GetChiralTag() in CHIRALITY_LIST
                            else 3,
                            FORMAL_CHARGE.index(atom.GetFormalCharge())
                            if atom.GetFormalCharge() in FORMAL_CHARGE
                            else 4,
                        ]
                    )

                x = torch.tensor(atom_feats, dtype=torch.long)

                scaffold_mask = self.get_scaffold(mol)

                edge_info = []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_feat = [
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                    ]
                    edge_info.append((start, end, edge_feat))
                    edge_info.append((end, start, edge_feat))

                edge_index = (
                    torch.tensor([(e[0], e[1]) for e in edge_info], dtype=torch.long)
                    .t()
                    .contiguous()
                )
                edge_attr = torch.tensor([e[2] for e in edge_info], dtype=torch.long)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    scaffold_mask=scaffold_mask,
                    num_nodes=x.size(0),
                    maccs=self.maccs_generator(mol),
                    ecfp4=self.ecfp4_generator(mol),
                )

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)

    @staticmethod
    def maccs_generator(mol):
        maccs = AllChem.GetMACCSKeysFingerprint(mol)
        return torch.tensor(list(map(int, maccs.ToBitString())), dtype=torch.float32)
    
    @staticmethod
    def ecfp4_generator(mol):
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return torch.tensor(list(map(int, ecfp4.ToBitString())), dtype=torch.float32)


class MaskSubgraph(BaseTransform):
    def __init__(self, mask_ratio: float, seed: int = 0) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.seed = seed

    def __call__(self, data: Data) -> tuple[Data, Data]:
        G = to_networkx(data)

        atom_scaffold_id = torch.where(data.scaffold_mask)[0].numpy()
        G, masked = self.masked_subgraph(G, atom_scaffold_id, self.mask_ratio, self.seed)
        # Fully connected between masked nodes
        i, j = torch.meshgrid(masked, masked, indexing="ij")
        masked_edge_index = torch.stack([i, j], dim=-1)

        # Remove self-loops
        self_loops = i == j
        masked_edge_index = masked_edge_index[~self_loops]

        # orginal edge_index
        edge_index = data.edge_index.clone().T

        comparison = masked_edge_index[:, None, :] == edge_index[None, :, :]
        matches = comparison.all(dim=2)
        complement_mask = ~(matches.any(dim=1))
        add_masked_index = masked_edge_index[complement_mask]
        
        if add_masked_index.size(0) != 0:
            new_edge_index = torch.cat([edge_index, add_masked_index], dim=0).to(torch.long)
            new_edge_attr = torch.cat([data.edge_attr.clone(), torch.tensor([[4, 3]] * add_masked_index.size(0))], dim=0).to(torch.long)
            edge_add_seg = torch.tensor([1] * edge_index.size(0) + [0] * add_masked_index.size(0), dtype=torch.bool)
        else:
            new_edge_index = edge_index.clone()
            new_edge_attr = data.edge_attr.clone()
            edge_add_seg = torch.tensor([1] * edge_index.size(0), dtype=torch.bool)
            
        for k in range(data.edge_attr.size(0)):
            if data.edge_index[0, k] in masked or data.edge_index[1, k] in masked:
                new_edge_attr[k] = torch.tensor([4, 3], dtype=torch.long)

        new_x = data.x.clone()
        new_x[masked] = torch.tensor([[len(ATOM_LIST), 4, 5]] * masked.size(0))
        masked_data = Data(
            x=new_x,
            edge_index=new_edge_index.T,
            edge_attr=new_edge_attr,
            edge_add_seg=edge_add_seg,
            num_nodes=data.num_nodes,
        )
        return data, masked_data

    @staticmethod
    def masked_subgraph(G: nx.Graph, atom_scaffold_id: np.ndarray, mask_ratio: float, seed: int = 0):
        assert mask_ratio <= 1
        masked_num = max(min(round(len(G.nodes) * mask_ratio), len(atom_scaffold_id)), 2)
        removed = []
        temp = np.random.choice(atom_scaffold_id, 1)

        while len(removed) < masked_num:
            neighbors = []
            for n in temp:
                neighbors.extend(
                    [
                        i
                        for i in G.neighbors(n)
                        if i not in temp and np.isin(atom_scaffold_id, i).any()
                    ]
                )
            np.random.default_rng(seed).shuffle(neighbors)
            for n in temp:
                if len(removed) < masked_num:
                    G.remove_node(n)
                    removed.append(n)
                else:
                    break
            temp = list(set(neighbors))
            if len(temp) == 0:
                break
        return G, torch.tensor(removed)


class MaskTransform(BaseTransform):
    def __init__(self, mask_ratio: float) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def __call__(self, inputs: Data) -> Data:
        data = inputs
        atom_scaffold = torch.where(data.scaffold_mask)[0]
        scaffold_num = torch.sum(data.scaffold_mask).item()
        num_samples = max(round(self.mask_ratio * scaffold_num), 1)
        atom_to_mask = atom_scaffold[torch.randperm(scaffold_num)[:num_samples]]

        x_masked = data.x.clone()
        # [9, 4, 5] is the mask embedding
        x_masked[atom_to_mask] = torch.tensor([len(ATOM_LIST), 4, 5], dtype=torch.long)

        edge_to_mask = self.find_neighbor_edge(
            atom_to_mask, data.edge_index, atom_scaffold
        )
        edge_attr_masked = data.edge_attr.clone()
        # [4, 3] is the mask embedding
        edge_attr_masked[edge_to_mask] = torch.tensor([4, 3], dtype=torch.long)

        data.x_masked = x_masked
        data.edge_attr_masked = edge_attr_masked
        return data

    def find_neighbor_edge(
        self, atom_idx: Tensor, edge_index: Tensor, scaffold: Tensor
    ):
        masked_edges = (edge_index[0] == atom_idx[:, None]) & torch.isin(
            edge_index[1], scaffold, assume_unique=False
        )
        return torch.where(masked_edges.any(dim=0))[0]


if __name__ == "__main__":
    dataset = MolGraphDataset(root="../../coconut_data", transform=MaskSubgraph(0.15))
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    print(next(iter(dataloader)))
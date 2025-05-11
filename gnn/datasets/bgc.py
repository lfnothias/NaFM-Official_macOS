import pickle
import numpy as np
from typing import Any
import pandas as pd
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
import csv
import ast

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

class BGC(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BGC, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ["bgc_data.csv"]

    @property
    def processed_file_names(self):
        return ["clean_mols.pt"]


    def process(self):
        error_count = 0 
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            df = pd.read_csv(path)
            smiles_list = df['smiles'].tolist()
            label_list = df['label'].tolist()
            bgc_ids = df['bgc_id'].tolist()

            samples = []
            
            for idx, smiles in tqdm(enumerate(smiles_list)):
                try:
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
                except:
                    print(f"Error in {smiles}")
                    error_count += 1
                    continue
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
                
                label = ast.literal_eval(label_list[idx])
                
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=x.size(0),
                    smiles=smiles,
                    BGC_id=bgc_ids[idx],
                    label=torch.tensor(label, dtype=torch.float).view(1, -1),
                    ecfp=self.ecfp4_generator(mol),
                )

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                samples.append(data)
        
            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)
        print(f"Error count: {error_count}")

    @staticmethod
    def ecfp4_generator(mol):
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return torch.tensor(list(map(int, ecfp4.ToBitString())), dtype=torch.float32)

    @property
    def num_class(self) -> int:
        return len(self.data.label[0])




if __name__ == "__main__":
    dataset = BGC(root="../../downstream_data/BGC_data")
    #print(dataset.data.BGC_id)
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=0,
    )
    for batch in dataloader:
        print(batch.label)
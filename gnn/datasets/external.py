import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import trange

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


class External(InMemoryDataset):
    def __init__(self, root, dataset_arg, transform=None, pre_transform=None):
        self.dataset_arg = dataset_arg

        super(External, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["external_data.csv"]

    @property
    def processed_file_names(self):
        return [f"External_{self.dataset_arg}.pt"]

    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            df = pd.read_csv(path)
            target_data = df.loc[df['target_id'] == f'NPT{self.dataset_arg}']
            smiles_list = target_data["SMILES"].tolist()
            labels = target_data["activity_label"].tolist()
            folds = target_data["fold"].tolist()
            coconut = target_data["is_coconut"].tolist()
            
            samples = []

            for i in trange(len(smiles_list)):
                label = labels[i]
                fold = folds[i]
                coconut_info = coconut[i]
                
                mol = Chem.MolFromSmiles(smiles_list[i])

                atom_feats = []
                for atom in mol.GetAtoms():
                    try:
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
                    except ValueError:
                        print(f"Error at index {i} with SMILES: {smiles_list[i]}")
                        raise

                x = torch.tensor(atom_feats, dtype=torch.long)

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
                    num_nodes=x.size(0),
                    label=torch.tensor(label, dtype=torch.long),
                    ecfp=self.ecfp4_generator(mol),
                    fold=torch.tensor(fold, dtype=torch.long),
                    # use true for coconut, false for non-coconut
                    is_coconut=torch.tensor(coconut_info, dtype=torch.bool),
                )

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)

    @staticmethod
    def ecfp4_generator(mol):
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return torch.tensor(list(map(int, ecfp4.ToBitString())), dtype=torch.float32)
    
    @property
    def num_class(self) -> int:
        return self.data.label.max().item() + 1


if __name__ == "__main__":
    dataset = External(
        root="../../downstream_data/External", dataset_arg="30"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    print(next(iter(dataloader)))
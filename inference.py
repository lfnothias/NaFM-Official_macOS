import argparse
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm, trange
from scipy.special import expit
from gnn.datasets import *
from gnn.tune_module import LNNP as FinetunedLNNP

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

def get_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        choices =['classification','regression','screen']
    )
    
    parser.add_argument(
        "--downstream-data",
        default=None,
        type=str,
        help="downstream data file",
    )

    parser.add_argument(
        "--num-classes",
        default=None,
        type=int,
        help="Number of classes for classification",
    )
    
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=str,
        help="checkpoint storage directory",
    )
    
    args = parser.parse_args()
    return args


def get_molgraph(smiles_list: list, labels: list):
    mol_graphs = []
    wrong_checks = 0
    for i in trange(len(smiles_list)):
        label = labels[i]
        mol = Chem.MolFromSmiles(smiles_list[i])
        if mol is None:
            print(f"Invalid SMILES at index {i}: {smiles_list[i]}")
            wrong_checks += 1
            continue  # 跳过无效的 SMILES
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
            torch.tensor([(e[0], e[1]) for e in edge_info], dtype=torch.long).t().contiguous()
                )
        edge_attr = torch.tensor([e[2] for e in edge_info], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=x.size(0),
            label=torch.tensor(label, dtype=torch.long),
        )

        mol_graphs.append(data)
    return mol_graphs, wrong_checks
        

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FinetunedLNNP.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        num_classes=args.num_classes,
        is_inference=True
    )
    model = model.to(device)
    model.eval()
    
    if args.task == 'screen':
        filter_data = pd.read_csv(args.downstream_data)
        to_filter = filter_data[filter_data['label'] == 0]
        smiles = to_filter['SMILES'].values
        labels = to_filter['label'].values
        # Generate molecular graphs
        data, wrong_checks = get_molgraph(smiles, labels)
        print(f"Number of wrong checks: {wrong_checks}")
    
        data_loader = DataLoader(data, batch_size=256, shuffle=False)
        scores = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                batch_data = batch_data.to(device)
                output = model(batch_data)
                scores.extend(output[:, 1].cpu().numpy())
            
            
        probabilities = expit(scores)
    
        # Combine SMILES and scores, save to a CSV file
        results_df = pd.DataFrame({'SMILES': smiles, 'Score': probabilities})
        results_df.sort_values('Score', ascending=False, inplace=True)
        results_df.to_csv('model_scores.csv', index=False)
        print("Results saved to model_scores.csv")
        
    elif args.task == 'classification':
        # Load the data
        data = pd.read_csv(args.downstream_data)
        smiles = data['SMILES'].values
        labels = [0 for _ in range(len(smiles))]
        # Generate molecular graphs
        data, wrong_checks = get_molgraph(smiles, labels)
        print(f"Number of wrong checks: {wrong_checks}")
        data_loader = DataLoader(data, batch_size=256, shuffle=False)
        
        # Perform inference
        all_preds = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                batch_data = batch_data.to(device)
                output = model(batch_data)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                all_preds.extend(preds)

        # Save the SMILES and predictions
        results_df = pd.DataFrame({'SMILES': smiles, 'Prediction': all_preds})
        results_df.to_csv('predictions.csv', index=False)
        print("Predictions saved to predictions.csv")
        
    elif args.task == 'regression':
        data = pd.read_csv(args.downstream_data)
        smiles = data['SMILES'].values
        labels = [0 for _ in range(len(smiles))]
        # Generate molecular graphs
        data, wrong_checks = get_molgraph(smiles, labels)
        print(f"Number of wrong checks: {wrong_checks}")
        data_loader = DataLoader(data, batch_size=256, shuffle=False)
        # Perform inference
        all_preds = []
        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                batch_data = batch_data.to(device)
                output = model(batch_data)
                preds = output.cpu().numpy()
                all_preds.extend(preds)
        # Save the SMILES and predictions
        results_df = pd.DataFrame({'SMILES': smiles, 'Prediction': all_preds})
        results_df.to_csv('predictions.csv', index=False)
        print("Predictions saved to predictions.csv")
        
    else:
        raise ValueError("Invalid task. Choose from 'classification', 'regression', or 'screen'.")
        
        
if __name__ == "__main__":
    main()
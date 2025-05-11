import pickle

import pandas as pd
from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

df = pd.read_csv("coconut_complete-10-2024.csv")
smiles_list = df['canonical_smiles'].tolist()

# Define a list of allowed atom symbols
# ATOM_SYMBOLS = ["H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
ATOM_SYMBOLS = ["C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]


# Function to standardize a SMILES string
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    # Try to create a molecule from the SMILES string
    if mol is None or Descriptors.MolWt(mol) > 2000:
        return 0

    # Check if the molecule has allowed atoms only and has more than one atom
    if (
        any(atom.GetSymbol() not in ATOM_SYMBOLS for atom in mol.GetAtoms())
        or mol.GetNumAtoms() == 1
    ):
        return 0

    # Standardize the molecule
    s = Standardizer()
    mol = s.standardize(mol)
    mol = s.fragment_parent(mol)

    # Remove salts from the molecule
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol, dontRemoveEverything=True)

    # Convert the molecule back to a SMILES string
    return Chem.MolToSmiles(mol)


# Standardize all SMILES strings in the list
with Pool(cpu_count()) as pool:
    clean_smiles = list(tqdm(pool.imap(standardize_smiles, smiles_list), total=len(smiles_list)))

# Remove duplicates and 0 values
clean_smiles = list(set(clean_smiles) - {0})

# Print the length of the cleaned SMILES list and check if it contains 0
print(len(clean_smiles))  # 405468
print(0 in clean_smiles)  # False

# Save the cleaned SMILES list to a pickle file
with open("clean_smiles.pkl", "wb") as f:
    pickle.dump(clean_smiles, f, protocol=4)
#!/usr/bin/env python3
"""
NaFM Data Setup Script
Automatically downloads and sets up required data files according to README requirements

Usage:
    python setup_data.py                    # Download all data
    python setup_data.py --skip-zenodo      # Skip pre-trained weights download
    python setup_data.py --skip-figshare    # Skip datasets download
    python setup_data.py --verify-only      # Only verify existing data
"""

import os
import sys
import requests
import zipfile
import tarfile
import hashlib
from pathlib import Path
import argparse
from tqdm import tqdm
import time

# Data source URLs
FIGSHARE_URL = "https://figshare.com/ndownloader/files/28980254"
ZENODO_URL = "https://zenodo.org/records/15385335/files/NaFM.ckpt"

# File structure as required by README
REQUIRED_FILES = {
    "raw_data/raw/pretrain_smiles.pkl": "Pre-training SMILES data",
    "downstream_data/Ontology/raw/classification_data.csv": "Ontology classification data",
    "downstream_data/Regression/raw/regression_data.csv": "Regression data",
    "downstream_data/Lotus/raw/lotus_data.csv": "Lotus data",
    "downstream_data/Bgc/raw/bgc_data.csv": "BGC data", 
    "downstream_data/External/raw/external_data.csv": "External data",
    "NaFM.ckpt": "Pre-trained model weights"
}

def create_directory_structure():
    """Create required directory structure"""
    print("Creating directory structure...")
    
    directories = [
        "raw_data/raw",
        "downstream_data/Ontology/raw",
        "downstream_data/Regression/raw",
        "downstream_data/Lotus/raw", 
        "downstream_data/Bgc/raw",
        "downstream_data/External/raw"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def download_file_with_progress(url, filepath, description="Downloading"):
    """Download file with progress bar"""
    print(f"Starting download: {description}")
    print(f"URL: {url}")
    print(f"Saving to: {filepath}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                    desc=description, ncols=80) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Download completed: {filepath}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False

def download_figshare_data():
    """Download datasets from Figshare"""
    print("\n" + "="*50)
    print("Downloading Figshare datasets")
    print("="*50)
    
    # Download zip file
    zip_path = "figshare_data.zip"
    if download_file_with_progress(FIGSHARE_URL, zip_path, "Figshare datasets"):
        # Extract files
        print("Extracting datasets...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("✓ Datasets extracted successfully")
            
            # Clean up zip file
            os.remove(zip_path)
            print("✓ Cleaned up temporary files")
            
        except zipfile.BadZipFile:
            print("❌ Corrupted zip file")
            return False
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    else:
        return False
    
    return True

def download_zenodo_weights():
    """Download pre-trained weights from Zenodo"""
    print("\n" + "="*50)
    print("Downloading Zenodo pre-trained weights")
    print("="*50)
    
    if download_file_with_progress(ZENODO_URL, "NaFM.ckpt", "Pre-trained weights"):
        # Verify file size
        file_size = os.path.getsize("NaFM.ckpt")
        print(f"✓ Weights file size: {file_size:,} bytes")
        return True
    else:
        return False

def verify_data_integrity():
    """Verify data integrity"""
    print("\n" + "="*50)
    print("Verifying data integrity")
    print("="*50)
    
    missing_files = []
    total_size = 0
    
    for filepath, description in REQUIRED_FILES.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            print(f"✓ {filepath} ({size:,} bytes) - {description}")
        else:
            print(f"❌ {filepath} - {description}")
            missing_files.append((filepath, description))
    
    print(f"\nTotal data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files:")
        for filepath, description in missing_files:
            print(f"  - {filepath} - {description}")
        return False
    else:
        print("\n✅ All required files are in place!")
        return True

def generate_checksums():
    """Generate file checksums"""
    print("\nGenerating file checksums...")
    checksums = {}
    
    for filepath in REQUIRED_FILES.keys():
        if os.path.exists(filepath):
            print(f"Calculating checksum for {filepath}...")
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                checksums[filepath] = file_hash.hexdigest()
    
    # Save checksums to file
    with open("checksums.txt", "w", encoding='utf-8') as f:
        f.write("# NaFM data file checksums\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for filepath, checksum in checksums.items():
            f.write(f"{checksum}  {filepath}\n")
    
    print("✓ Checksums saved to checksums.txt")
    return checksums

def print_next_steps():
    """Print next steps instructions"""
    print("\n" + "="*50)
    print("🎉 Data setup completed!")
    print("="*50)
    print("\nNext steps:")
    print("1. Pre-train model:")
    print("   python train.py --conf examples/Pretrain.yml")
    print("\n2. Fine-tune model:")
    print("   python train.py --conf examples/Finetune.yml")
    print("\n3. Run inference:")
    print("   python inference.py --task classification \\")
    print("     --downstream-data your_data.csv \\")
    print("     --checkpoint-path your_model.ckpt")
    print("\n4. Verify setup:")
    print("   python validate_setup.py")

def main():
    parser = argparse.ArgumentParser(description="NaFM data setup script")
    parser.add_argument("--skip-figshare", action="store_true", 
                       help="Skip Figshare data download")
    parser.add_argument("--skip-zenodo", action="store_true",
                       help="Skip Zenodo weights download")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing data, don't download")
    
    args = parser.parse_args()
    
    print("NaFM Data Setup Script")
    print("="*50)
    print("Setting up required data files according to README requirements")
    print("="*50)
    
    if not args.verify_only:
        # Create directory structure
        create_directory_structure()
        
        # Download data
        success = True
        
        if not args.skip_figshare:
            if not download_figshare_data():
                success = False
        
        if not args.skip_zenodo:
            if not download_zenodo_weights():
                success = False
        
        if not success:
            print("\n❌ Errors occurred during download, please check network connection and retry")
            sys.exit(1)
    
    # Verify data integrity
    if verify_data_integrity():
        generate_checksums()
        print_next_steps()
    else:
        print("\n❌ Data setup incomplete, please check missing files")
        print("Tip: If some files failed to download, you can download them manually and place in correct locations")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NaFM Setup Validation Script
Checks environment, dependencies, data integrity and basic functionality

Usage:
    python validate_setup.py
"""

import os
import sys
import importlib
import subprocess
import platform
from pathlib import Path
import torch
import pytorch_lightning as pl

def check_python_version():
    """Check Python version compatibility"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
        return False

def check_dependencies():
    """Check all required dependency packages"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'torch', 'pytorch_lightning', 'torch_geometric', 'rdkit', 
        'numpy', 'pandas', 'scikit-learn', 'networkx', 'scipy',
        'tqdm', 'tensorboard', 'torchmetrics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_torch_setup():
    """Check PyTorch installation and CUDA availability"""
    print("\nChecking PyTorch setup...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return True

def check_data_files():
    """Check if all required data files exist"""
    print("\nChecking data files...")
    
    required_files = {
        "raw_data/raw/pretrain_smiles.pkl": "Pre-training SMILES data",
        "downstream_data/Ontology/raw/classification_data.csv": "Ontology data",
        "downstream_data/Regression/raw/regression_data.csv": "Regression data",
        "downstream_data/Lotus/raw/lotus_data.csv": "Lotus data", 
        "downstream_data/Bgc/raw/bgc_data.csv": "BGC data",
        "downstream_data/External/raw/external_data.csv": "External data",
        "NaFM.ckpt": "Pre-trained weights"
    }
    
    missing_files = []
    total_size = 0
    
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            print(f"✓ {filepath} ({size:,} bytes) - {description}")
        else:
            print(f"❌ {filepath} - {description}")
            missing_files.append(filepath)
    
    print(f"\nTotal data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("Run: python setup_data.py to download missing data")
        return False
    
    return True

def check_model_loading():
    """Test if pre-trained model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        checkpoint_path = "NaFM.ckpt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("✓ Pre-trained model checkpoint loaded successfully")
            
            # Check if it's a Lightning checkpoint
            if 'state_dict' in checkpoint:
                print("✓ Lightning checkpoint format detected")
            else:
                print("⚠️  Checkpoint format may not be Lightning-compatible")
            
            return True
        else:
            print("❌ Pre-trained model checkpoint not found")
            return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def run_basic_tests():
    """Run basic functionality tests"""
    print("\nRunning basic tests...")
    
    try:
        # Test basic imports
        from gnn.data import PretrainedDataModule, FinetunedDataModule
        from gnn.pre_module import LNNP as PretrainedLNNP
        from gnn.tune_module import LNNP as FinetunedLNNP
        print("✓ Core modules imported successfully")
        
        # Test data module creation
        if os.path.exists("raw_data/raw/pretrain_smiles.pkl"):
            data_module = PretrainedDataModule(
                hparams=dict(
                    dataset_root="raw_data",
                    batch_size=32,
                    num_workers=0
                )
            )
            print("✓ Data module created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False

def main():
    """Run all validation checks"""
    print("NaFM Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies), 
        ("PyTorch Setup", check_torch_setup),
        ("Data Files", check_data_files),
        ("Model Loading", check_model_loading),
        ("Basic Functionality", run_basic_tests)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your NaFM setup is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please address the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Convenience script to run setup and validation from project root
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

def main():
    """Run setup and validation scripts"""
    import setup_data
    import validate_setup
    
    print("NaFM Setup and Validation")
    print("=" * 50)
    
    # Run setup first
    print("Step 1: Setting up data...")
    setup_data.main()
    
    print("\n" + "=" * 50)
    print("Step 2: Validating setup...")
    validate_setup.main()

if __name__ == "__main__":
    main()

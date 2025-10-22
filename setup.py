#!/usr/bin/env python3
"""
NaFM Setup Script - Convenience entry point
Runs setup and validation scripts from the scripts/ directory
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

def main():
    """Run setup and validation"""
    print("NaFM Complete Setup")
    print("=" * 50)
    print("This will download data and validate your setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("scripts/setup_data.py"):
        print("❌ Error: Please run this script from the NaFM project root directory")
        print("Current directory:", os.getcwd())
        return 1
    
    # Run setup first
    print("\nStep 1: Setting up data...")
    try:
        import setup_data
        setup_data.main()
        print("\n✓ Data setup completed successfully!")
    except SystemExit as e:
        if e.code != 0:
            print(f"\n❌ Data setup failed with exit code {e.code}")
            return e.code
    except ImportError as e:
        print(f"\n❌ Failed to import setup_data: {e}")
        print("Please ensure you're in the correct directory and have the required dependencies")
        return 1
    except Exception as e:
        print(f"\n❌ Data setup failed with error: {e}")
        return 1
    
    # Run validation
    print("\n" + "=" * 50)
    print("Step 2: Validating setup...")
    try:
        import validate_setup
        result = validate_setup.main()
        if result == 0:
            print("\n🎉 Setup and validation completed successfully!")
            print("\nYou can now use NaFM:")
            print("  python train.py --conf examples/Pretrain.yml")
            print("  python train.py --conf examples/Finetune.yml")
            print("  python inference.py --help")
        return result
    except ImportError as e:
        print(f"\n❌ Failed to import validate_setup: {e}")
        print("Please ensure you have the required dependencies installed")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

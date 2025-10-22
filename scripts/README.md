# NaFM Scripts

This directory contains utility scripts for NaFM setup and validation.

## Available Scripts

### Core Scripts

- **`setup_data.py`** - Automated data download and organization
  - Downloads data from Figshare and Zenodo
  - Creates proper directory structure
  - Generates file checksums
  - Usage: `python scripts/setup_data.py [options]`

- **`validate_setup.py`** - Comprehensive setup validation
  - Checks Python version compatibility
  - Verifies all dependencies
  - Validates data file integrity
  - Tests model loading capability
  - Usage: `python scripts/validate_setup.py`

- **`run_setup.py`** - Combined setup and validation
  - Runs both setup and validation in sequence
  - Usage: `python scripts/run_setup.py`

## Quick Start

From the project root directory:

```bash
# Complete setup and validation
python setup.py

# Or run scripts individually
python scripts/setup_data.py
python scripts/validate_setup.py
```

## Script Options

### setup_data.py Options

```bash
python scripts/setup_data.py --help

Options:
  --skip-figshare    Skip Figshare data download
  --skip-zenodo      Skip Zenodo weights download  
  --verify-only      Only verify existing data, don't download
```

### validate_setup.py

No options required - runs comprehensive validation checks.

## Output Files

- `checksums.txt` - File integrity checksums
- Data files in proper directory structure
- Validation report in console output

## Troubleshooting

If scripts fail:

1. Check network connectivity for downloads
2. Verify Python environment and dependencies
3. Run `python scripts/validate_setup.py` to diagnose issues
4. Check file permissions in the project directory

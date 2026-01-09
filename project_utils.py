"""
MTH Project Utility Script

This script demonstrates various utility functions for working with the
industrial network analysis project. It can be used for common tasks like
data inspection, model evaluation, and configuration management.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check if all required packages are installed."""
    print("Checking environment...")
    print("-" * 50)
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'tensorflow',
        'matplotlib',
        'plotly',
        'dash',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    print("-" * 50)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True


def show_project_structure():
    """Display the project directory structure."""
    print("\nProject Structure:")
    print("-" * 50)
    
    try:
        from config_manager import config
        print(config)
    except ImportError:
        print("Config manager not available. Using basic structure:")
        print(f"Project root: {project_root}")


def validate_data_files():
    """Validate available data files."""
    print("\nValidating Data Files:")
    print("-" * 50)
    
    try:
        from data_validator import DataValidator
        from config_manager import config
        
        validator = DataValidator()
        
        # Check if data directory exists
        if not config.DATA_DIR.exists():
            print(f"✗ Data directory not found: {config.DATA_DIR}")
            return False
        
        print(f"✓ Data directory exists: {config.DATA_DIR}")
        
        # Check for device data file
        device_file = config.get_device_data_path()
        if device_file.exists():
            print(f"✓ Device data file found: {device_file.name}")
            
            # Quick validation
            is_valid, report = validator.validate_csv_file(str(device_file))
            if is_valid:
                print("✓ File is valid")
            else:
                print("✗ File validation failed:")
                print(report)
        else:
            print(f"✗ Device data file not found: {device_file}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import required modules: {e}")
        return False


def list_available_data():
    """List all CSV files in the data directory."""
    print("\nAvailable Data Files:")
    print("-" * 50)
    
    try:
        from config_manager import config
        
        if not config.DATA_DIR.exists():
            print("Data directory not found")
            return
        
        csv_files = list(config.DATA_DIR.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in data directory")
        else:
            for i, file in enumerate(csv_files, 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"{i}. {file.name} ({size_mb:.2f} MB)")
        
    except ImportError:
        print("Config manager not available")


def show_model_info():
    """Display information about available models."""
    print("\nModel Information:")
    print("-" * 50)
    
    try:
        from config_manager import config
        
        # Check forecasting model
        if config.FORECASTING_MODEL_DIR.exists():
            print(f"✓ Forecasting model directory: {config.FORECASTING_MODEL_DIR}")
            model_files = list(config.FORECASTING_MODEL_DIR.glob("*"))
            if model_files:
                print(f"  Files: {len(model_files)}")
        else:
            print(f"✗ Forecasting model directory not found")
        
        # Check classification model
        if config.CLASSIFICATION_MODEL_DIR.exists():
            print(f"✓ Classification model directory: {config.CLASSIFICATION_MODEL_DIR}")
            
            model_file = config.get_classification_model_path()
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"  Model file: {model_file.name} ({size_mb:.2f} MB)")
            
            scaler_file = config.get_classification_scaler_path()
            if scaler_file.exists():
                print(f"  Scaler file: {scaler_file.name}")
            
            encoder_file = config.get_classification_encoders_path()
            if encoder_file.exists():
                print(f"  Encoder file: {encoder_file.name}")
        else:
            print(f"✗ Classification model directory not found")
            
    except ImportError:
        print("Config manager not available")


def create_sample_config():
    """Create a sample configuration file."""
    print("\nCreating Sample Configuration:")
    print("-" * 50)
    
    try:
        from config_manager import config
        
        config_file = project_root / "config.json"
        config.save_to_file(config_file)
        print(f"✓ Sample configuration saved to: {config_file}")
        print("\nYou can edit this file to customize paths for your system.")
        
    except ImportError:
        print("Config manager not available")


def run_diagnostics():
    """Run comprehensive diagnostics on the project."""
    print("\n" + "=" * 60)
    print(" MTH PROJECT DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Check environment
    env_ok = check_environment()
    print()
    
    # 2. Show project structure
    show_project_structure()
    print()
    
    # 3. List available data
    list_available_data()
    print()
    
    # 4. Show model info
    show_model_info()
    print()
    
    # Summary
    print("\n" + "=" * 60)
    print(" DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if env_ok:
        print("✓ Environment setup is complete")
    else:
        print("✗ Environment setup needs attention")
        print("  Run: pip install -r requirements.txt")
    
    print("\nFor more information, see CODE_ANALYSIS.md")


def main():
    """Main entry point for the utility script."""
    parser = argparse.ArgumentParser(
        description="MTH Project Utility Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_utils.py --check-env        Check if all packages are installed
  python project_utils.py --validate-data    Validate data files
  python project_utils.py --list-data        List available data files
  python project_utils.py --model-info       Show model information
  python project_utils.py --create-config    Create sample config file
  python project_utils.py --diagnostics      Run full diagnostics
        """
    )
    
    parser.add_argument('--check-env', action='store_true',
                       help='Check if required packages are installed')
    parser.add_argument('--validate-data', action='store_true',
                       help='Validate data files')
    parser.add_argument('--list-data', action='store_true',
                       help='List available data files')
    parser.add_argument('--model-info', action='store_true',
                       help='Show model information')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Run comprehensive diagnostics')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute requested actions
    if args.check_env:
        check_environment()
    
    if args.validate_data:
        validate_data_files()
    
    if args.list_data:
        list_available_data()
    
    if args.model_info:
        show_model_info()
    
    if args.create_config:
        create_sample_config()
    
    if args.diagnostics:
        run_diagnostics()


if __name__ == "__main__":
    main()

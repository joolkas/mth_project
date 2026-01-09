"""
Example Usage of MTH Project Utilities

This script demonstrates how to use the new utility modules added to the project.
Run this after installing dependencies with: pip install -r requirements.txt
"""

# Example 1: Using the Configuration Manager
print("=" * 60)
print("Example 1: Configuration Manager")
print("=" * 60)

try:
    from config_manager import config
    
    print("\nConfiguration loaded successfully!")
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Default device: {config.DEFAULT_DEVICE}")
    
    # Get specific paths
    device_path = config.get_device_data_path()
    print(f"\nDevice data path: {device_path}")
    
    forecasting_path = config.get_processed_forecasting_path()
    print(f"Forecasting data path: {forecasting_path}")
    
    # Create necessary directories
    print("\nCreating required directories...")
    config.ensure_directories_exist()
    print("✓ Directories created/verified")
    
except ImportError as e:
    print(f"✗ Could not import config_manager: {e}")


# Example 2: Using the Data Validator
print("\n" + "=" * 60)
print("Example 2: Data Validator")
print("=" * 60)

try:
    import pandas as pd
    import numpy as np
    from data_validator import DataValidator
    
    # Create sample network data
    print("\nCreating sample network data...")
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='min'),
        'Interface GigabitEthernet1/1: Bits Sent': np.random.randint(0, 1000000, 100),
        'Interface GigabitEthernet1/1: Bits Received': np.random.randint(0, 1000000, 100),
        'Interface GigabitEthernet1/2: Bits Sent': np.random.randint(0, 1000000, 100),
        'Interface GigabitEthernet1/2: Bits Received': np.random.randint(0, 1000000, 100),
        'CPU Usage': np.random.uniform(20, 80, 100),
        'Temperature': np.random.uniform(20, 30, 100),
    })
    
    sample_data = sample_data.set_index('timestamp')
    print(f"Sample data shape: {sample_data.shape}")
    
    # Validate the data
    validator = DataValidator()
    
    print("\nValidating DataFrame structure...")
    if validator.validate_network_dataframe(sample_data):
        print("✓ DataFrame is valid")
    else:
        print("✗ DataFrame validation failed")
    
    # Show any warnings
    if validator.warnings:
        print("\nValidation warnings:")
        for warning in validator.warnings:
            print(f"  - {warning}")
    
    # Categorize columns
    print("\nCategorizing columns...")
    categories = validator.validate_network_columns(sample_data)
    for category, columns in categories.items():
        if columns:
            print(f"  {category}: {len(columns)} columns")
    
    # Get quality report
    print("\nData quality report:")
    quality = validator.check_data_quality(sample_data)
    print(f"  Total rows: {quality['total_rows']}")
    print(f"  Total columns: {quality['total_columns']}")
    print(f"  Missing values: {quality['missing_values']}")
    print(f"  Missing percentage: {quality['missing_percentage']:.2f}%")
    print(f"  Duplicate rows: {quality['duplicate_rows']}")
    
    if quality['constant_columns']:
        print(f"  Constant columns: {len(quality['constant_columns'])}")
    
    # Validate time series
    print("\nValidating time series properties...")
    if validator.validate_time_series(sample_data):
        print("✓ Valid time series data")
    else:
        print("✗ Time series validation issues found")
    
except ImportError as e:
    print(f"✗ Could not import required modules: {e}")
    print("  Install dependencies with: pip install -r requirements.txt")


# Example 3: Integrating with existing code
print("\n" + "=" * 60)
print("Example 3: Integration Pattern")
print("=" * 60)

print("""
Here's how to integrate these utilities into existing code:

# Old way (hardcoded paths):
data_path = "C:\\\\ThesisWork\\\\...\\\\Data092025\\\\"
model_path = "C:\\\\ThesisWork\\\\...\\\\classification_model\\\\"

# New way (portable):
from config_manager import config
data_path = config.DATA_DIR
model_path = config.CLASSIFICATION_MODEL_DIR

# Add data validation before processing:
from data_validator import DataValidator
import pandas as pd

validator = DataValidator()
df = pd.read_csv(data_file)

if validator.validate_network_dataframe(df):
    # Process data
    processed_df = preprocess(df)
else:
    print("Data validation failed:")
    print(validator.get_report())
""")


# Example 4: Using with real data (if available)
print("\n" + "=" * 60)
print("Example 4: Working with Real Data (if available)")
print("=" * 60)

try:
    from config_manager import config
    from data_validator import quick_validate_file
    import pandas as pd
    
    device_file = config.get_device_data_path()
    
    if device_file.exists():
        print(f"\nFound data file: {device_file.name}")
        
        # Quick validation
        is_valid, report = quick_validate_file(str(device_file))
        
        if is_valid:
            print("✓ File is valid")
            
            # Try to load first few rows
            print("\nLoading sample data (first 5 rows)...")
            df_sample = pd.read_csv(device_file, nrows=5)
            print(df_sample.head())
            
        else:
            print("✗ File validation failed")
            print(report)
    else:
        print(f"\n✗ Data file not found: {device_file}")
        print("   This is expected if you're running on a different machine.")
        print("   The utilities still work with portable paths!")
        
except ImportError:
    print("✗ Required modules not available")
except Exception as e:
    print(f"✗ Error: {e}")


# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
New utilities provided:

1. config_manager.py - Centralized, portable path management
   • Cross-platform compatibility
   • Easy directory structure management
   • JSON configuration support

2. data_validator.py - Data quality checking
   • CSV file validation
   • DataFrame structure validation
   • Column categorization
   • Quality metrics

3. project_utils.py - Command-line utilities
   • Environment checking
   • Data validation
   • Model information
   • Diagnostics

4. requirements.txt - Dependency management
   • Easy installation: pip install -r requirements.txt

5. CODE_ANALYSIS.md - Comprehensive code review
   • Identified issues and solutions
   • Best practices recommendations
   • Security considerations

For more details, see UTILITIES_README.md
""")

print("\nTo run diagnostics on your setup:")
print("  python project_utils.py --diagnostics")

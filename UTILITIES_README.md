# MTH Project Utilities

This document describes the new utility files added to improve code quality, portability, and usability of the MTH project.

## New Files Added

### 1. `CODE_ANALYSIS.md`
Comprehensive code analysis report covering:
- Code structure and strengths
- Issues and recommendations (Critical, Medium, and Low priority)
- Security considerations
- Performance considerations
- Suggestions for new programs

**Key findings:**
- Hardcoded Windows paths need to be fixed
- Missing import in main.py
- Bug in dataset.py (fixed)
- No requirements.txt (now added)

### 2. `requirements.txt`
Dependencies file for easy installation:
```bash
pip install -r requirements.txt
```

Contains all necessary packages:
- pandas, numpy, scikit-learn
- tensorflow
- matplotlib, plotly, dash
- scipy, jupyter

### 3. `config_manager.py`
Centralized configuration management to replace hardcoded paths.

**Features:**
- Cross-platform path handling using `pathlib`
- Centralized directory management
- JSON config file support
- Helper methods for common paths

**Usage:**
```python
from config_manager import config

# Get device data path
data_path = config.get_device_data_path("A1-SW-B-246")

# Get model paths
model_path = config.get_classification_model_path()

# Create directories if needed
config.ensure_directories_exist()
```

### 4. `data_validator.py`
Data validation utility for checking data quality before processing.

**Features:**
- CSV file validation
- DataFrame structure validation
- Column categorization (bits sent/received, status, etc.)
- Data quality metrics
- Time series validation

**Usage:**
```python
from data_validator import DataValidator, quick_validate_file

# Quick validation
is_valid, report = quick_validate_file("data.csv")
print(report)

# Detailed validation
validator = DataValidator()
if validator.validate_network_dataframe(df):
    print("Data is valid!")
    
# Get quality metrics
quality = validator.check_data_quality(df)
print(f"Missing values: {quality['missing_percentage']:.2f}%")
```

### 5. `project_utils.py`
Command-line utility for common project tasks.

**Features:**
- Check environment (installed packages)
- Validate data files
- List available data
- Show model information
- Create sample config
- Run diagnostics

**Usage:**
```bash
# Check if all packages installed
python project_utils.py --check-env

# Validate data files
python project_utils.py --validate-data

# List available data
python project_utils.py --list-data

# Show model info
python project_utils.py --model-info

# Run full diagnostics
python project_utils.py --diagnostics

# Create sample config
python project_utils.py --create-config
```

## Bug Fixes

### Fixed in `mth_project/dataset.py`
**Issue:** Line 31 called `.lower()` on a list instead of individual strings, causing AttributeError.

**Before:**
```python
for port in df_network_traffic['name'].unique().lower():  # Bug!
    port_number = port.split('gi1/')[1].split('(')[0].strip()
```

**After:**
```python
for port in df_network_traffic['name'].unique():
    port_lower = port.lower()
    if 'gi1/' not in port_lower:
        continue
    try:
        port_number = port_lower.split('gi1/')[1].split('(')[0].strip()
        # ... with proper error handling
    except (IndexError, AttributeError) as e:
        print(f"Warning: Could not parse port from {port}: {e}")
        continue
```

## How to Use These Utilities

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run diagnostics to check setup:**
   ```bash
   python project_utils.py --diagnostics
   ```

3. **Validate your data:**
   ```bash
   python project_utils.py --validate-data
   ```

### Migrating Existing Code

To make existing code portable, replace hardcoded paths with config manager:

**Before:**
```python
data_path = "C:\\ThesisWork\\offical_approach\\mth_project\\..."
```

**After:**
```python
from config_manager import config
data_path = config.get_device_data_path()
```

### Validation Before Processing

Add validation before processing data:

```python
from data_validator import DataValidator

validator = DataValidator()
if validator.validate_csv_file(filepath):
    # Proceed with processing
    df = pd.read_csv(filepath)
    if validator.validate_network_dataframe(df):
        # Data is good, continue
        process_data(df)
else:
    print(validator.get_report())
```

## Benefits

1. **Portability:** Code now works on Windows, Linux, and Mac
2. **Maintainability:** Centralized configuration, easier to update
3. **Reliability:** Data validation catches issues early
4. **Usability:** Utility scripts for common tasks
5. **Documentation:** Clear analysis of code issues and solutions

## Next Steps

To fully integrate these utilities into the existing codebase:

1. Update `get_data.py` to use `config_manager`
2. Update `classification_model.py` to use `config_manager`
3. Update `main.py` to use `config_manager`
4. Add data validation calls before processing
5. Create unit tests for core functions
6. Add docstrings to functions without documentation

See `CODE_ANALYSIS.md` for detailed recommendations.

## Testing the Utilities

### Test config_manager
```bash
python config_manager.py
```

### Test data_validator
```bash
python data_validator.py
```

### Test project_utils
```bash
python project_utils.py --diagnostics
```

## Support

For issues or questions about these utilities, refer to:
- `CODE_ANALYSIS.md` - Detailed code analysis
- Individual file docstrings - API documentation
- README.md - Project overview

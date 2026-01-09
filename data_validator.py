"""
Data Validation Utility for MTH Project

This module provides utilities to validate network data files before processing,
helping catch issues early and provide clear error messages.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DataValidator:
    """
    Validates industrial network data files and DataFrames.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.errors = []
        self.warnings = []
    
    def clear_messages(self):
        """Clear all error and warning messages."""
        self.errors = []
        self.warnings = []
    
    def validate_csv_file(self, filepath: str) -> bool:
        """
        Validate a CSV file exists and is readable.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            bool: True if valid, False otherwise
        """
        self.clear_messages()
        
        # Check file exists
        if not Path(filepath).exists():
            self.errors.append(f"File not found: {filepath}")
            return False
        
        # Check file is not empty
        if Path(filepath).stat().st_size == 0:
            self.errors.append(f"File is empty: {filepath}")
            return False
        
        # Try to read file - check first few rows and a sample from middle
        try:
            # Check header and first rows
            df_head = pd.read_csv(filepath, nrows=5)
            if df_head.empty:
                self.errors.append(f"File contains no data: {filepath}")
                return False
            
            # Try to get rough row count and sample from middle for better validation
            try:
                # Read a larger sample to catch issues that might appear later
                df_sample = pd.read_csv(filepath, nrows=100)
                if len(df_sample) < len(df_head):
                    self.warnings.append("File has fewer than 100 rows")
            except Exception:
                # If we can't read more, that's ok - we already validated the header
                pass
                
        except Exception as e:
            self.errors.append(f"Error reading file {filepath}: {str(e)}")
            return False
        
        return True
    
    def validate_network_dataframe(self, df: pd.DataFrame, 
                                   check_columns: Optional[List[str]] = None) -> bool:
        """
        Validate a network data DataFrame has expected structure.
        
        Args:
            df: DataFrame to validate
            check_columns: Optional list of required column patterns
            
        Returns:
            bool: True if valid, False otherwise
        """
        self.clear_messages()
        
        # Check DataFrame is not None
        if df is None:
            self.errors.append("DataFrame is None")
            return False
        
        # Check DataFrame is not empty
        if df.empty:
            self.errors.append("DataFrame is empty")
            return False
        
        # Check for required column patterns
        if check_columns:
            for pattern in check_columns:
                matching = [col for col in df.columns if pattern.lower() in col.lower()]
                if not matching:
                    self.errors.append(f"No columns found matching pattern: {pattern}")
        
        # Check for timestamp column
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        if not timestamp_cols and 'timestamp' not in str(df.index.name).lower():
            self.warnings.append("No timestamp column found")
        
        # Check for NaN values
        nan_percentage = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if nan_percentage > 50:
            self.warnings.append(f"High percentage of NaN values: {nan_percentage:.2f}%")
        elif nan_percentage > 0:
            self.warnings.append(f"DataFrame contains {nan_percentage:.2f}% NaN values")
        
        # Check data types
        non_numeric = df.select_dtypes(exclude=['number', 'datetime64']).columns
        if len(non_numeric) > 0 and 'timestamp' not in str(df.index.name).lower():
            self.warnings.append(f"Non-numeric columns found: {list(non_numeric)[:5]}")
        
        return len(self.errors) == 0
    
    def validate_network_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify and categorize network data columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict with categorized column names
        """
        categories = {
            'bits_sent': [],
            'bits_received': [],
            'operational_status': [],
            'temperature': [],
            'cpu': [],
            'memory': [],
            'icmp': [],
            'other': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if 'bits sent' in col_lower:
                categories['bits_sent'].append(col)
            elif 'bits received' in col_lower:
                categories['bits_received'].append(col)
            elif 'operational status' in col_lower:
                categories['operational_status'].append(col)
            elif 'temperature' in col_lower:
                categories['temperature'].append(col)
            elif 'cpu' in col_lower:
                categories['cpu'].append(col)
            elif 'memory' in col_lower:
                categories['memory'].append(col)
            elif 'icmp' in col_lower:
                categories['icmp'].append(col)
            else:
                categories['other'].append(col)
        
        return categories
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isna().sum().sum(),
            'missing_percentage': (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'constant_columns': [],
            'high_cardinality_columns': [],
        }
        
        # Check for constant columns
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if df[col].nunique() <= 1:
                    quality_report['constant_columns'].append(col)
        
        # Check for high cardinality columns (potential issues)
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                quality_report['high_cardinality_columns'].append(col)
        
        return quality_report
    
    def validate_time_series(self, df: pd.DataFrame) -> bool:
        """
        Validate time series data properties.
        
        Args:
            df: DataFrame with datetime index or timestamp column
            
        Returns:
            bool: True if valid time series
        """
        self.clear_messages()
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find timestamp column
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
            if not timestamp_cols:
                self.errors.append("No datetime index or timestamp column found")
                return False
            else:
                self.warnings.append("Timestamp found as column, not as index")
        
        # Check for time gaps
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()
            if len(time_diffs) > 1:
                median_diff = time_diffs.median()
                max_gap = time_diffs.max()
                
                if max_gap > median_diff * 10:
                    self.warnings.append(f"Large time gaps detected. Median: {median_diff}, Max: {max_gap}")
        
        # Check for duplicate timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.duplicated().any():
                dup_count = df.index.duplicated().sum()
                self.warnings.append(f"Duplicate timestamps found: {dup_count}")
        
        return len(self.errors) == 0
    
    def get_report(self) -> str:
        """
        Get a formatted report of all errors and warnings.
        
        Returns:
            str: Formatted report
        """
        report = []
        
        if self.errors:
            report.append("ERRORS:")
            for i, error in enumerate(self.errors, 1):
                report.append(f"  {i}. {error}")
        
        if self.warnings:
            report.append("\nWARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            report.append("No errors or warnings found.")
        
        return "\n".join(report)


def quick_validate_file(filepath: str) -> Tuple[bool, str]:
    """
    Quick validation of a data file.
    
    Args:
        filepath: Path to file to validate
        
    Returns:
        Tuple of (is_valid, report_message)
    """
    validator = DataValidator()
    
    if not validator.validate_csv_file(filepath):
        return False, validator.get_report()
    
    try:
        df = pd.read_csv(filepath, nrows=100)
        validator.validate_network_dataframe(df, check_columns=['bits', 'timestamp'])
        return len(validator.errors) == 0, validator.get_report()
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


if __name__ == "__main__":
    print("Data Validation Utility for MTH Project")
    print("=" * 50)
    
    # Example usage
    validator = DataValidator()
    
    # Create sample data for demonstration
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='min'),
        'Interface GigabitEthernet1/1: Bits Sent': np.random.randint(0, 1000000, 100),
        'Interface GigabitEthernet1/1: Bits Received': np.random.randint(0, 1000000, 100),
        'Temperature': np.random.uniform(20, 30, 100),
    })
    
    sample_df = sample_df.set_index('timestamp')
    
    print("\nValidating sample DataFrame...")
    if validator.validate_network_dataframe(sample_df):
        print("✓ DataFrame is valid")
    else:
        print("✗ DataFrame validation failed")
    
    print("\n" + validator.get_report())
    
    print("\n\nColumn categorization:")
    categories = validator.validate_network_columns(sample_df)
    for category, columns in categories.items():
        if columns:
            print(f"  {category}: {len(columns)} columns")
    
    print("\n\nData quality report:")
    quality = validator.check_data_quality(sample_df)
    for key, value in quality.items():
        print(f"  {key}: {value}")

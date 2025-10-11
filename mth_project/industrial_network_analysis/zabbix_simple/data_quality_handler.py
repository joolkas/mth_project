#!/usr/bin/env python3
"""
Data Quality Handler for Industrial Network Analysis
Implements Good/Bad flagging system and last-value interpolation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

class DataQualityHandler:
    """
    Handles data quality flagging and last-value interpolation
    
    Key features:
    1. Flags samples as "Good" (real from Zabbix) or "Bad" (missing/suspicious)
    2. For "Bad" samples: Replaces with last known good value
    3. For "Good" samples: Keeps original Zabbix value
    4. Maintains quality statistics for monitoring
    """
    
    def __init__(self, variables: list, max_cache_age_minutes: int = 30):
        """
        Initialize data quality handler
        
        Args:
            variables: List of variable names to monitor
            max_cache_age_minutes: Maximum age of cached values in minutes
        """
        self.variables = variables
        self.max_cache_age = timedelta(minutes=max_cache_age_minutes)
        
        # Cache for last known good values
        self.last_good_values = {}  # {variable_name: {'value': float, 'timestamp': datetime, 'quality': 'Good'}}
        
        # Quality statistics
        self.quality_stats = {var: {'good_count': 0, 'bad_count': 0, 'interpolated_count': 0} for var in variables}
        self.total_requests = 0
        
        logger.info(f"DataQualityHandler initialized for {len(variables)} variables")
        logger.info(f"Maximum cache age: {max_cache_age_minutes} minutes")
    
    def process_data_with_quality_flags(self, raw_data: pd.DataFrame, 
                                       target_length: int,
                                       target_frequency: str = '1min') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process raw Zabbix data with quality flagging and interpolation
        
        Args:
            raw_data: Raw data from Zabbix (may have gaps or zeros)
            target_length: Desired number of data points
            target_frequency: Target frequency (e.g., '1min' for 1 minute)
            
        Returns:
            Tuple of (processed_data, quality_flags)
            - processed_data: DataFrame with interpolated values
            - quality_flags: DataFrame with 'Good'/'Bad' flags for each sample
        """
        self.total_requests += 1
        
        logger.info(f"Processing data with quality flags - Request #{self.total_requests}")
        
        # Create target time range
        if raw_data is not None and not raw_data.empty:
            end_time = raw_data.index[-1]
        else:
            end_time = datetime.now()
        
        start_time = end_time - timedelta(minutes=target_length)
        
        # Create complete time index
        target_index = pd.date_range(start=start_time, end=end_time, freq=target_frequency)
        
        # Initialize output DataFrames
        processed_data = pd.DataFrame(index=target_index, columns=self.variables)
        quality_flags = pd.DataFrame(index=target_index, columns=self.variables)
        
        logger.info(f"Target time range: {start_time} to {end_time} ({len(target_index)} points)")
        
        # Process each variable
        for variable in self.variables:
            logger.debug(f"Processing variable: {variable}")
            
            processed_values, quality_values = self._process_variable_quality(
                variable, raw_data, target_index
            )
            
            processed_data[variable] = processed_values
            quality_flags[variable] = quality_values
        
        # Log quality statistics
        self._log_quality_statistics()
        
        return processed_data, quality_flags
    
    def _process_variable_quality(self, variable: str, raw_data: pd.DataFrame, 
                                 target_index: pd.DatetimeIndex) -> Tuple[List[float], List[str]]:
        """
        Process a single variable with quality flagging
        
        Args:
            variable: Variable name
            raw_data: Raw Zabbix data
            target_index: Target time index
            
        Returns:
            Tuple of (processed_values, quality_flags)
        """
        processed_values = []
        quality_flags = []
        
        for timestamp in target_index:
            value, quality = self._get_value_with_quality(variable, raw_data, timestamp)
            processed_values.append(value)
            quality_flags.append(quality)
            
            # Update statistics
            if quality == 'Good':
                self.quality_stats[variable]['good_count'] += 1
            else:
                self.quality_stats[variable]['bad_count'] += 1
                if quality == 'Interpolated':
                    self.quality_stats[variable]['interpolated_count'] += 1
        
        return processed_values, quality_flags
    
    def _get_value_with_quality(self, variable: str, raw_data: pd.DataFrame, 
                               timestamp: datetime) -> Tuple[float, str]:
        """
        Get value for a specific timestamp with quality assessment
        
        Args:
            variable: Variable name
            raw_data: Raw Zabbix data
            timestamp: Target timestamp
            
        Returns:
            Tuple of (value, quality_flag)
            - quality_flag: 'Good', 'Bad', 'Interpolated', 'Default'
        """
        
        # Case 1: Check if we have exact data for this timestamp
        if (raw_data is not None and not raw_data.empty and 
            variable in raw_data.columns and timestamp in raw_data.index):
            
            raw_value = raw_data.loc[timestamp, variable]
            
            if not pd.isna(raw_value):
                # Assess if this is a "Good" value
                if self._is_good_value(variable, raw_value, timestamp):
                    # Update cache with good value
                    self._update_cache(variable, raw_value, timestamp)
                    return float(raw_value), 'Good'
                else:
                    # Value exists but is "Bad" (e.g., suspicious zero)
                    interpolated_value = self._get_interpolated_value(variable, timestamp)
                    return interpolated_value, 'Bad'
        
        # Case 2: No exact data - look for recent data within reasonable time window
        if raw_data is not None and not raw_data.empty and variable in raw_data.columns:
            recent_data = raw_data[raw_data.index <= timestamp][variable].dropna()
            
            if len(recent_data) > 0:
                recent_timestamp = recent_data.index[-1]
                time_gap = timestamp - recent_timestamp
                
                # If recent data is within 5 minutes, consider using it
                if time_gap <= timedelta(minutes=5):
                    recent_value = recent_data.iloc[-1]
                    
                    if self._is_good_value(variable, recent_value, recent_timestamp):
                        # Use recent good value but flag as interpolated
                        self._update_cache(variable, recent_value, recent_timestamp)
                        return float(recent_value), 'Interpolated'
        
        # Case 3: Use cached last known good value
        interpolated_value = self._get_interpolated_value(variable, timestamp)
        return interpolated_value, 'Interpolated'
    
    def _is_good_value(self, variable: str, value: float, timestamp: datetime) -> bool:
        """
        Assess if a value is "Good" quality
        
        Args:
            variable: Variable name
            value: Value to assess
            timestamp: Timestamp of the value
            
        Returns:
            True if value is considered good quality
        """
        # Basic checks
        if pd.isna(value) or np.isinf(value):
            return False
        
        # For network variables, be more careful about zeros
        if self._is_network_variable(variable):
            # Single zero might be legitimate (no traffic)
            # But check if this creates a suspicious pattern
            if value == 0:
                # Check cache to see if we have a recent non-zero value
                if variable in self.last_good_values:
                    cached_data = self.last_good_values[variable]
                    cache_age = timestamp - cached_data['timestamp']
                    
                    # If we had non-zero traffic recently, this zero might be suspicious
                    if (cache_age <= timedelta(minutes=10) and 
                        cached_data['value'] > 0):
                        logger.debug(f"Suspicious zero detected for {variable}: recent cache had {cached_data['value']}")
                        return False  # Flag as bad
                
                # Otherwise, accept the zero as potentially legitimate
                return True
            else:
                # Non-zero network values are generally good
                return True
        else:
            # For non-network variables (CPU, memory), zeros are more suspicious
            if value == 0:
                logger.debug(f"Zero value flagged as suspicious for non-network variable {variable}")
                return False
            return True
    
    def _is_network_variable(self, variable_name: str) -> bool:
        """Check if a variable represents network interface data"""
        network_indicators = ['bits', 'bytes', 'packets', 'interface', 'network', 'traffic']
        variable_lower = variable_name.lower()
        return any(indicator in variable_lower for indicator in network_indicators)
    
    def _update_cache(self, variable: str, value: float, timestamp: datetime) -> None:
        """Update cache with a good value"""
        self.last_good_values[variable] = {
            'value': value,
            'timestamp': timestamp,
            'quality': 'Good'
        }
        logger.debug(f"Updated cache for {variable}: {value} at {timestamp}")
    
    def _get_interpolated_value(self, variable: str, timestamp: datetime) -> float:
        """Get interpolated value (last known good value or default)"""
        
        # Try to use cached last known good value
        if variable in self.last_good_values:
            cached_data = self.last_good_values[variable]
            cache_age = timestamp - cached_data['timestamp']
            
            if cache_age <= self.max_cache_age:
                logger.debug(f"Using cached value for {variable}: {cached_data['value']} (age: {cache_age})")
                return cached_data['value']
            else:
                logger.warning(f"Cached value for {variable} too old ({cache_age}), using default")
        
        # Default values based on variable type
        if self._is_network_variable(variable):
            default_value = 1000.0  # Small positive value for network variables
        else:
            default_value = 1.0  # Small positive value for other variables
        
        logger.debug(f"Using default value for {variable}: {default_value}")
        return default_value
    
    def _log_quality_statistics(self) -> None:
        """Log quality statistics periodically"""
        if self.total_requests % 10 == 0:  # Log every 10 requests
            logger.info("Data Quality Statistics:")
            logger.info(f"  Total requests: {self.total_requests}")
            
            for variable in self.variables:
                stats = self.quality_stats[variable]
                total = stats['good_count'] + stats['bad_count']
                if total > 0:
                    good_pct = (stats['good_count'] / total) * 100
                    interpolated_pct = (stats['interpolated_count'] / total) * 100
                    logger.info(f"  {variable}: {good_pct:.1f}% good, {interpolated_pct:.1f}% interpolated")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary for monitoring"""
        summary = {
            'total_requests': self.total_requests,
            'cached_variables': len(self.last_good_values),
            'variable_stats': {}
        }
        
        for variable in self.variables:
            stats = self.quality_stats[variable]
            total = stats['good_count'] + stats['bad_count']
            
            var_summary = {
                'good_count': stats['good_count'],
                'bad_count': stats['bad_count'],
                'interpolated_count': stats['interpolated_count'],
                'good_percentage': (stats['good_count'] / total * 100) if total > 0 else 0,
                'has_cache': variable in self.last_good_values
            }
            
            if variable in self.last_good_values:
                cached_data = self.last_good_values[variable]
                var_summary['cached_value'] = cached_data['value']
                var_summary['cache_age_minutes'] = (datetime.now() - cached_data['timestamp']).total_seconds() / 60
            
            summary['variable_stats'][variable] = var_summary
        
        return summary


def test_data_quality_handler():
    """Test the DataQualityHandler with sample data"""
    print("Testing DataQualityHandler...")
    
    # Create test variables
    variables = [
        'cpu_usage', 
        'memory_usage', 
        'network_bits_in', 
        'network_bits_out'
    ]
    
    handler = DataQualityHandler(variables, max_cache_age_minutes=30)
    
    # Create test data with missing values and suspicious zeros
    dates = pd.date_range('2025-10-11 19:28:00', periods=15, freq='1min')
    raw_data = pd.DataFrame({
        'cpu_usage': [5.0, 5.1, np.nan, 5.2, 4.8, np.nan, 4.9, 5.1, 4.8, 5.0, np.nan, 5.2, 4.8, 5.0, 4.9],
        'memory_usage': [60.0, 60.5, 61.0, np.nan, 61.2, 60.9, np.nan, 60.7, 61.0, 60.8, 61.2, np.nan, 61.1, 60.7, 61.0],
        'network_bits_in': [41264, 42000, 43500, 0, 0, 0, np.nan, np.nan, 44500, 45000, 45500, 0, 0, 46000, 46500],  # Mixed zeros and NaN
        'network_bits_out': [46920, 47000, 47500, 47200, np.nan, 0, 0, 0, np.nan, 48000, 48200, 48500, 48800, 0, 49000]  # Mixed pattern
    }, index=dates)
    
    print(f"Raw data sample (network_bits_in): {raw_data['network_bits_in'].tolist()}")
    
    # Process with quality handler
    processed_data, quality_flags = handler.process_data_with_quality_flags(
        raw_data=raw_data,
        target_length=20,
        target_frequency='1min'
    )
    
    print(f"Processed data sample (network_bits_in): {processed_data['network_bits_in'].tail(10).tolist()}")
    print(f"Quality flags sample (network_bits_in): {quality_flags['network_bits_in'].tail(10).tolist()}")
    
    # Show quality summary
    summary = handler.get_quality_summary()
    print(f"Quality summary: {summary}")
    
    return processed_data, quality_flags


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_data_quality_handler()
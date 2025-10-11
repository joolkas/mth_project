#!/usr/bin/env python3
"""
Smart Data Handler for Industrial Network Analysis
Handles missing data by maintaining last-known-good values instead of zero-filling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SmartDataBuffer:
    """
    Smart data buffer that handles missing Zabbix data intelligently
    Instead of filling with zeros, it maintains last-known-good values for each variable
    """
    
    def __init__(self, variables: list, max_cache_age_minutes: int = 30):
        """
        Initialize smart data buffer
        
        Args:
            variables: List of variable names to monitor
            max_cache_age_minutes: Maximum age of cached values in minutes
        """
        self.variables = variables
        self.max_cache_age = timedelta(minutes=max_cache_age_minutes)
        
        # Cache for last known good values
        self.last_good_values = {}  # {variable_name: {'value': float, 'timestamp': datetime}}
        self.missing_data_counts = {var: 0 for var in variables}
        self.total_requests = 0
        
        logger.info(f"SmartDataBuffer initialized for {len(variables)} variables")
        logger.info(f"Maximum cache age: {max_cache_age_minutes} minutes")
    
    def update_cache(self, df: pd.DataFrame, timestamp: Optional[datetime] = None) -> None:
        """
        Update the cache with fresh data from Zabbix
        
        Args:
            df: DataFrame with fresh data from Zabbix
            timestamp: Timestamp for the data (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if df is None or df.empty:
            logger.warning("Empty data provided to update_cache")
            return
        
        # Update cache with the most recent valid values
        for variable in self.variables:
            if variable in df.columns:
                # Get the most recent non-zero, non-NaN value
                recent_values = df[variable].dropna()
                
                if len(recent_values) > 0:
                    # For network interface data, we need to be careful about legitimate zeros
                    if self._is_network_variable(variable):
                        # For network variables, only reject obvious missing data patterns
                        # Keep legitimate zero values (no traffic periods)
                        valid_values = recent_values[recent_values >= 0]  # Accept zeros for network
                        
                        if len(valid_values) > 0:
                            latest_value = float(valid_values.iloc[-1])
                            self.last_good_values[variable] = {
                                'value': latest_value,
                                'timestamp': timestamp
                            }
                            logger.debug(f"Updated cache for {variable}: {latest_value}")
                    else:
                        # For non-network variables (CPU, memory), prefer non-zero values
                        non_zero_values = recent_values[recent_values > 0]
                        
                        if len(non_zero_values) > 0:
                            latest_value = float(non_zero_values.iloc[-1])
                        else:
                            # If all values are zero, use the most recent one
                            latest_value = float(recent_values.iloc[-1])
                        
                        self.last_good_values[variable] = {
                            'value': latest_value,
                            'timestamp': timestamp
                        }
                        logger.debug(f"Updated cache for {variable}: {latest_value}")
    
    def _is_network_variable(self, variable_name: str) -> bool:
        """Check if a variable represents network interface data"""
        network_indicators = ['bits', 'bytes', 'packets', 'interface', 'network', 'traffic']
        variable_lower = variable_name.lower()
        return any(indicator in variable_lower for indicator in network_indicators)
    
    def get_smart_filled_data(self, df: pd.DataFrame, target_length: int, 
                             target_frequency: str = '1T') -> pd.DataFrame:
        """
        Fill missing data points with last-known-good values instead of zeros
        
        Args:
            df: Raw data from Zabbix (may have gaps)
            target_length: Desired number of data points
            target_frequency: Target frequency (e.g., '1T' for 1 minute)
            
        Returns:
            DataFrame with missing points filled using last-known-good values
        """
        self.total_requests += 1
        
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided - using cached values only")
            return self._create_from_cache_only(target_length, target_frequency)
        
        # Update cache with fresh data
        self.update_cache(df)
        
        # Create target time range
        end_time = df.index[-1] if len(df) > 0 else datetime.now()
        start_time = end_time - timedelta(minutes=target_length)
        
        # Create complete time index
        target_index = pd.date_range(start=start_time, end=end_time, freq=target_frequency)
        target_df = pd.DataFrame(index=target_index, columns=self.variables)
        
        logger.info(f"Creating smart-filled data: {len(target_index)} points from {start_time} to {end_time}")
        
        # Fill data intelligently for each variable
        for variable in self.variables:
            filled_count = 0
            
            # Start with original data where available
            if variable in df.columns:
                # Map original data to target index
                for timestamp in target_index:
                    # Look for exact match first
                    if timestamp in df.index and not pd.isna(df.loc[timestamp, variable]):
                        target_df.loc[timestamp, variable] = df.loc[timestamp, variable]
                    else:
                        # Look for closest previous data point within reasonable range
                        recent_data = df[df.index <= timestamp][variable].dropna()
                        
                        if len(recent_data) > 0:
                            # Use most recent value within last 10 minutes
                            recent_timestamp = recent_data.index[-1]
                            if timestamp - recent_timestamp <= timedelta(minutes=10):
                                target_df.loc[timestamp, variable] = recent_data.iloc[-1]
                            else:
                                # Too old, use cached value
                                cached_value = self._get_cached_value(variable)
                                if cached_value is not None:
                                    target_df.loc[timestamp, variable] = cached_value
                                    filled_count += 1
                        else:
                            # No recent data, use cached value
                            cached_value = self._get_cached_value(variable)
                            if cached_value is not None:
                                target_df.loc[timestamp, variable] = cached_value
                                filled_count += 1
            else:
                # Variable not in current data, use cached values
                cached_value = self._get_cached_value(variable)
                if cached_value is not None:
                    target_df[variable] = cached_value
                    filled_count = len(target_index)
                else:
                    # No cached value available, use safe default
                    if self._is_network_variable(variable):
                        target_df[variable] = 0  # Legitimate zero for network interfaces
                    else:
                        target_df[variable] = 1  # Safe non-zero default for other metrics
                    filled_count = len(target_index)
                    logger.warning(f"No cached value for {variable}, using default")
            
            if filled_count > 0:
                self.missing_data_counts[variable] += filled_count
                logger.debug(f"Filled {filled_count} missing points for {variable} using smart logic")
        
        # Final cleanup
        target_df = target_df.fillna(method='ffill').fillna(method='bfill')
        target_df = target_df.fillna(0)  # Final fallback only
        
        # Convert to numeric types
        for col in target_df.columns:
            target_df[col] = pd.to_numeric(target_df[col], errors='coerce').fillna(0)
        
        logger.info(f"Smart data filling completed: {target_df.shape}")
        self._log_statistics()
        
        return target_df
    
    def _get_cached_value(self, variable: str) -> Optional[float]:
        """Get cached value for a variable if not too old"""
        if variable not in self.last_good_values:
            return None
        
        cached_data = self.last_good_values[variable]
        cache_age = datetime.now() - cached_data['timestamp']
        
        if cache_age <= self.max_cache_age:
            return cached_data['value']
        else:
            logger.warning(f"Cached value for {variable} too old ({cache_age}), not using")
            return None
    
    def _create_from_cache_only(self, target_length: int, target_frequency: str) -> pd.DataFrame:
        """Create DataFrame using only cached values when no fresh data available"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=target_length)
        target_index = pd.date_range(start=start_time, end=end_time, freq=target_frequency)
        
        target_df = pd.DataFrame(index=target_index, columns=self.variables)
        
        for variable in self.variables:
            cached_value = self._get_cached_value(variable)
            if cached_value is not None:
                target_df[variable] = cached_value
            else:
                # Use safe defaults
                if self._is_network_variable(variable):
                    target_df[variable] = 0
                else:
                    target_df[variable] = 1
        
        logger.warning(f"Created DataFrame from cache only: {target_df.shape}")
        return target_df
    
    def _log_statistics(self):
        """Log statistics about data filling"""
        if self.total_requests % 10 == 0:  # Log every 10 requests
            logger.info("SmartDataBuffer Statistics:")
            logger.info(f"  Total requests: {self.total_requests}")
            logger.info(f"  Cached variables: {len(self.last_good_values)}")
            
            total_missing = sum(self.missing_data_counts.values())
            if total_missing > 0:
                logger.info(f"  Total missing data points filled: {total_missing}")
                
                # Show top variables with missing data
                sorted_missing = sorted(self.missing_data_counts.items(), 
                                      key=lambda x: x[1], reverse=True)
                for var, count in sorted_missing[:3]:
                    if count > 0:
                        logger.info(f"    {var}: {count} points")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get detailed cache status for monitoring"""
        status = {
            'total_variables': len(self.variables),
            'cached_variables': len(self.last_good_values),
            'total_requests': self.total_requests,
            'missing_data_total': sum(self.missing_data_counts.values()),
            'cache_details': {}
        }
        
        for variable in self.variables:
            if variable in self.last_good_values:
                cached_data = self.last_good_values[variable]
                cache_age = datetime.now() - cached_data['timestamp']
                status['cache_details'][variable] = {
                    'value': cached_data['value'],
                    'age_minutes': cache_age.total_seconds() / 60,
                    'missing_filled': self.missing_data_counts[variable]
                }
            else:
                status['cache_details'][variable] = {
                    'value': None,
                    'age_minutes': None,
                    'missing_filled': self.missing_data_counts[variable]
                }
        
        return status


def test_smart_data_handler():
    """Test the SmartDataBuffer functionality"""
    print("Testing SmartDataBuffer...")
    
    # Create test variables
    variables = ['cpu_usage', 'memory_usage', 'network_bits_in', 'network_bits_out']
    buffer = SmartDataBuffer(variables, max_cache_age_minutes=30)
    
    # Create test data with gaps
    dates = pd.date_range('2025-01-01 10:00:00', periods=10, freq='2T')  # 2-minute intervals
    test_data = pd.DataFrame({
        'cpu_usage': [10, 15, 20, np.nan, 25, 30, np.nan, 35, 40, 45],
        'memory_usage': [50, 55, np.nan, 65, 70, np.nan, 80, 85, 90, 95],
        'network_bits_in': [1000, 2000, 0, 1500, np.nan, 3000, 0, 2500, 1800, 2200],
        'network_bits_out': [500, 800, 0, np.nan, 1200, 1500, 0, 1300, 900, 1100]
    }, index=dates)
    
    print(f"Original test data:\n{test_data}")
    
    # Test smart filling
    filled_data = buffer.get_smart_filled_data(test_data, target_length=20, target_frequency='1T')
    
    print(f"\nSmart-filled data:\n{filled_data}")
    print(f"\nCache status:\n{buffer.get_cache_status()}")
    
    # Test with completely missing data
    print("\nTesting with no fresh data (cache only)...")
    cache_only_data = buffer.get_smart_filled_data(pd.DataFrame(), target_length=5, target_frequency='1T')
    print(f"Cache-only data:\n{cache_only_data}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_smart_data_handler()
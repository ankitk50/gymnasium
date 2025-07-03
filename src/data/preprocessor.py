"""
Data preprocessing utilities.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Optional, List, Union
import torch


class DataPreprocessor:
    """
    Data preprocessing utilities for CPU allocation datasets.
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method: Scaling method ('standard', 'minmax', 'robust', 'none')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method != 'none':
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            feature_names: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features
            feature_names: Names of features
            
        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Fit imputer for missing values
        self.imputer = SimpleImputer(strategy='mean')
        X_imputed = self.imputer.fit_transform(X)
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Impute missing values
        X_imputed = self.imputer.transform(X)
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        return X_scaled
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                     feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            X: Features to fit and transform
            feature_names: Names of features
            
        Returns:
            Transformed features
        """
        return self.fit(X, feature_names).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X: Scaled features
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        if self.scaler is not None:
            return self.scaler.inverse_transform(X)
        else:
            return X
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the fitted features.
        
        Returns:
            Feature statistics dictionary
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted to get statistics")
        
        stats = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'scaling_method': self.scaling_method
        }
        
        if self.scaler is not None:
            if hasattr(self.scaler, 'mean_'):
                stats['feature_means'] = self.scaler.mean_.tolist()
            if hasattr(self.scaler, 'scale_'):
                stats['feature_scales'] = self.scaler.scale_.tolist()
            if hasattr(self.scaler, 'data_min_'):
                stats['feature_mins'] = self.scaler.data_min_.tolist()
            if hasattr(self.scaler, 'data_max_'):
                stats['feature_maxs'] = self.scaler.data_max_.tolist()
        
        return stats


def create_temporal_features(df: pd.DataFrame, 
                           timestamp_column: str,
                           value_columns: List[str],
                           window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Create temporal features from time series data.
    
    Args:
        df: Input DataFrame
        timestamp_column: Name of timestamp column
        value_columns: Columns to create temporal features for
        window_sizes: Window sizes for rolling statistics
        
    Returns:
        DataFrame with temporal features
    """
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column)
    
    for col in value_columns:
        for window in window_sizes:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            
            # Rolling std
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
            
            # Rolling min/max
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Difference features
        df[f'{col}_diff_1'] = df[col].diff(1)
        df[f'{col}_diff_2'] = df[col].diff(2)
    
    return df


def engineer_cpu_allocation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific features for CPU allocation.
    
    Args:
        df: Input DataFrame with basic features
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ['task_priority', 'memory_requirement', 'computation_complexity']
    for col in required_cols:
        if col not in df.columns:
            # Create dummy column if missing
            df[col] = np.random.rand(len(df))
    
    # Priority-weighted complexity
    df['priority_weighted_complexity'] = df['task_priority'] * df['computation_complexity']
    
    # Resource pressure score
    if 'memory_requirement' in df.columns and 'computation_complexity' in df.columns:
        df['resource_pressure'] = df['memory_requirement'] * df['computation_complexity']
    
    # Task urgency score
    if 'deadline_urgency' in df.columns and 'task_priority' in df.columns:
        df['urgency_score'] = df['deadline_urgency'] * df['task_priority']
    
    # Efficiency ratio
    if 'historical_runtime' in df.columns and 'computation_complexity' in df.columns:
        df['efficiency_ratio'] = df['computation_complexity'] / np.clip(df['historical_runtime'], 0.1, None)
    
    # Contention adjustment
    if 'resource_contention' in df.columns:
        df['contention_penalty'] = 1.0 - df['resource_contention']
        df['adjusted_priority'] = df['task_priority'] * df['contention_penalty']
    
    # Workload category features (if workload_type exists)
    if 'workload_type' in df.columns:
        # One-hot encode workload types
        workload_dummies = pd.get_dummies(df['workload_type'], prefix='workload')
        df = pd.concat([df, workload_dummies], axis=1)
    
    return df


def detect_outliers(X: np.ndarray, method: str = 'iqr', 
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in the data.
    
    Args:
        X: Input features
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (X < lower_bound) | (X > upper_bound)
        return np.any(outliers, axis=1)
    
    elif method == 'zscore':
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        return np.any(z_scores > threshold, axis=1)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def create_data_splits(X: np.ndarray, y: np.ndarray,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Create stratified data splits for training, validation, and testing.
    
    Args:
        X: Features
        y: Targets
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), 
        random_state=random_state, stratify=None
    )
    
    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_test_ratio),
        random_state=random_state, stratify=None
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

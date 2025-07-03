"""
Data loading utilities for the training pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class CPUAllocationDataset(Dataset):
    """
    Dataset for CPU allocation tasks.
    """
    
    def __init__(self, 
                 data: Union[pd.DataFrame, np.ndarray, str, Path],
                 target_column: str = 'cpu_allocation',
                 feature_columns: Optional[List[str]] = None,
                 sequence_length: Optional[int] = None,
                 task_type: str = 'regression',
                 transform: Optional[object] = None):
        """
        Initialize the CPU allocation dataset.
        
        Args:
            data: Data source (DataFrame, array, or file path)
            target_column: Name of target column
            feature_columns: List of feature column names
            sequence_length: Length for sequence data (for LSTM/Transformer)
            task_type: Type of task ('regression' or 'classification')
            transform: Data transformation function
        """
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.task_type = task_type
        self.transform = transform
        
        # Load and process data
        self.data = self._load_data(data)
        self.features, self.targets = self._prepare_data()
        
    def _load_data(self, data: Union[pd.DataFrame, np.ndarray, str, Path]) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            # Convert numpy array to DataFrame with default column names
            n_cols = data.shape[1]
            columns = [f'feature_{i}' for i in range(n_cols - 1)] + [self.target_column]
            return pd.DataFrame(data, columns=columns)
        elif isinstance(data, (str, Path)):
            # Load from file
            file_path = Path(data)
            if file_path.suffix == '.csv':
                return pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare features and targets."""
        # Select feature columns
        if self.feature_columns is None:
            # Use all columns except target
            self.feature_columns = [col for col in self.data.columns if col != self.target_column]
        
        # Extract features and targets
        X = self.data[self.feature_columns].values.astype(np.float32)
        y = self.data[self.target_column].values
        
        # Handle target based on task type
        if self.task_type == 'classification':
            # Encode labels for classification
            le = LabelEncoder()
            y = le.fit_transform(y).astype(np.int64)
        else:
            y = y.astype(np.float32)
        
        # Convert to tensors
        features = torch.from_numpy(X)
        targets = torch.from_numpy(y)
        
        # Handle sequence data
        if self.sequence_length is not None:
            features, targets = self._create_sequences(features, targets)
        
        return features, targets
    
    def _create_sequences(self, features: torch.Tensor, 
                         targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for temporal models."""
        n_samples = len(features)
        n_sequences = n_samples - self.sequence_length + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Sequence length {self.sequence_length} is too long for data with {n_samples} samples")
        
        # Create sequences
        seq_features = []
        seq_targets = []
        
        for i in range(n_sequences):
            seq_features.append(features[i:i + self.sequence_length])
            seq_targets.append(targets[i + self.sequence_length - 1])  # Use last target in sequence
        
        return torch.stack(seq_features), torch.stack(seq_targets)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        features = self.features[idx]
        targets = self.targets[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, targets


def create_synthetic_cpu_allocation_data(n_samples: int = 10000,
                                       n_features: int = 10,
                                       task_type: str = 'regression',
                                       sequence_length: Optional[int] = None,
                                       noise_level: float = 0.1) -> pd.DataFrame:
    """
    Create synthetic CPU allocation dataset for testing and demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        task_type: Type of task ('regression' or 'classification')
        sequence_length: If specified, create temporal dependencies
        noise_level: Amount of noise to add
        
    Returns:
        Synthetic dataset as DataFrame
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate base features
    features = np.random.rand(n_samples, n_features)
    
    # Define feature meanings for CPU allocation
    feature_names = [
        'task_priority', 'memory_requirement', 'io_intensity', 
        'computation_complexity', 'deadline_urgency', 'dependency_count',
        'historical_runtime', 'resource_contention', 'user_priority', 'workload_type'
    ][:n_features]
    
    # Pad with generic names if needed
    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')
    
    # Create realistic CPU allocation logic
    def compute_cpu_allocation(row):
        """Compute CPU allocation based on feature values."""
        priority_weight = row[0] * 0.3  # task_priority
        complexity_weight = row[3] * 0.25 if len(row) > 3 else 0  # computation_complexity
        urgency_weight = row[4] * 0.2 if len(row) > 4 else 0  # deadline_urgency
        memory_weight = row[1] * 0.15 if len(row) > 1 else 0  # memory_requirement
        contention_penalty = row[7] * -0.1 if len(row) > 7 else 0  # resource_contention
        
        base_allocation = (priority_weight + complexity_weight + 
                          urgency_weight + memory_weight + contention_penalty)
        
        # Add some non-linear relationships
        if len(row) > 2:  # io_intensity
            base_allocation += np.sin(row[2] * np.pi) * 0.1
        
        # Clamp to valid range
        return np.clip(base_allocation, 0.1, 1.0)
    
    # Generate targets
    if task_type == 'regression':
        targets = np.apply_along_axis(compute_cpu_allocation, 1, features)
        # Add noise
        targets += np.random.normal(0, noise_level, n_samples)
        targets = np.clip(targets, 0.0, 1.0)
        target_name = 'cpu_allocation'
    else:  # classification
        continuous_targets = np.apply_along_axis(compute_cpu_allocation, 1, features)
        # Convert to discrete classes (low, medium, high allocation)
        targets = np.digitize(continuous_targets, bins=[0.33, 0.66]) 
        target_name = 'cpu_allocation_class'
    
    # Add temporal dependencies if requested
    if sequence_length is not None:
        # Create temporal correlations
        for i in range(1, n_samples):
            # Current allocation influenced by previous allocation
            if task_type == 'regression':
                temporal_influence = targets[i-1] * 0.3
                targets[i] = 0.7 * targets[i] + 0.3 * temporal_influence
            
            # Add some temporal patterns to features
            features[i] += features[i-1] * 0.1  # Slight correlation with previous
    
    # Create DataFrame
    data_dict = {name: features[:, i] for i, name in enumerate(feature_names)}
    data_dict[target_name] = targets
    
    return pd.DataFrame(data_dict)


def create_data_loaders(data_config: Dict[str, Any],
                       task_type: str = 'regression') -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_config: Data configuration dictionary
        task_type: Type of task ('regression' or 'classification')
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get data source
    data_source = data_config.get('source', 'synthetic')
    
    if data_source == 'synthetic':
        # Create synthetic data
        n_samples = data_config.get('n_samples', 10000)
        n_features = data_config.get('n_features', 10)
        sequence_length = data_config.get('sequence_length')
        noise_level = data_config.get('noise_level', 0.1)
        
        df = create_synthetic_cpu_allocation_data(
            n_samples=n_samples,
            n_features=n_features,
            task_type=task_type,
            sequence_length=sequence_length,
            noise_level=noise_level
        )
    else:
        # Load from file
        data_path = data_config.get('path')
        if not data_path:
            raise ValueError("Data path must be specified when source is not 'synthetic'")
        df = pd.read_csv(data_path)
    
    # Data preprocessing
    if data_config.get('normalize', True):
        feature_columns = [col for col in df.columns if col not in ['cpu_allocation', 'cpu_allocation_class']]
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # Create dataset
    target_column = 'cpu_allocation' if task_type == 'regression' else 'cpu_allocation_class'
    feature_columns = data_config.get('feature_columns')
    sequence_length = data_config.get('sequence_length')
    
    dataset = CPUAllocationDataset(
        data=df,
        target_column=target_column,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        task_type=task_type
    )
    
    # Split data
    train_ratio = data_config.get('train_ratio', 0.7)
    val_ratio = data_config.get('val_ratio', 0.15)
    test_ratio = data_config.get('test_ratio', 0.15)
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # Calculate split sizes
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Split dataset
    train_dataset, temp_dataset = random_split(dataset, [n_train, n_total - n_train])
    
    if n_val > 0:
        val_dataset, test_dataset = random_split(temp_dataset, [n_val, n_test])
    else:
        val_dataset = None
        test_dataset = temp_dataset
    
    # Create data loaders
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) if test_dataset is not None else None
    
    return train_loader, val_loader, test_loader


class DataTransform:
    """Data transformation utilities."""
    
    @staticmethod
    def normalize(data: torch.Tensor, mean: Optional[torch.Tensor] = None, 
                 std: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalize data to zero mean and unit variance."""
        if mean is None:
            mean = data.mean(dim=0, keepdim=True)
        if std is None:
            std = data.std(dim=0, keepdim=True)
        
        return (data - mean) / (std + 1e-8)
    
    @staticmethod
    def min_max_scale(data: torch.Tensor, min_val: float = 0.0, 
                     max_val: float = 1.0) -> torch.Tensor:
        """Scale data to specified range."""
        data_min = data.min(dim=0, keepdim=True)[0]
        data_max = data.max(dim=0, keepdim=True)[0]
        
        normalized = (data - data_min) / (data_max - data_min + 1e-8)
        return normalized * (max_val - min_val) + min_val
    
    @staticmethod
    def add_noise(data: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to data."""
        noise = torch.randn_like(data) * noise_level
        return data + noise

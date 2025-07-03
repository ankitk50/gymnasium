"""
Configuration management utilities.
"""

import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json


class Config:
    """
    Configuration management class that supports YAML and dictionary formats.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or {}
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to JSON configuration file
            
        Returns:
            Config instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.learning_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self._config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., 'model.learning_rate')
            value: Value to set
        """
        keys = key.split('.')
        current = self._config
        
        # Navigate to the parent of the final key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    def update(self, other_config: Union['Config', Dict[str, Any]]) -> None:
        """
        Update configuration with another configuration.
        
        Args:
            other_config: Another configuration to merge
        """
        if isinstance(other_config, Config):
            self._deep_update(self._config, other_config._config)
        else:
            self._deep_update(self._config, other_config)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Update dictionary
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            file_path: Path to save YAML file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def save_json(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def copy(self) -> 'Config':
        """
        Create a copy of the configuration.
        
        Returns:
            Copy of configuration
        """
        return Config(self.to_dict())
    
    def validate(self, schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema.
        
        Args:
            schema: Validation schema
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        return self._validate_recursive(self._config, schema, "")
    
    def _validate_recursive(self, config: Dict[str, Any], 
                          schema: Dict[str, Any], path: str) -> bool:
        """
        Recursively validate configuration.
        
        Args:
            config: Configuration to validate
            schema: Schema to validate against
            path: Current path for error reporting
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        for key, schema_value in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in config:
                if schema_value.get('required', False):
                    raise ValueError(f"Required key '{current_path}' missing from configuration")
                continue
            
            config_value = config[key]
            expected_type = schema_value.get('type')
            
            if expected_type and not isinstance(config_value, expected_type):
                raise ValueError(f"Key '{current_path}' expected type {expected_type.__name__}, "
                               f"got {type(config_value).__name__}")
            
            # Validate nested dictionaries
            if isinstance(schema_value, dict) and 'type' not in schema_value:
                if not isinstance(config_value, dict):
                    raise ValueError(f"Key '{current_path}' expected dict, got {type(config_value).__name__}")
                self._validate_recursive(config_value, schema_value, current_path)
        
        return True
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style setting."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.get(key) is not None
    
    def __str__(self) -> str:
        """String representation."""
        return yaml.dump(self._config, default_flow_style=False, indent=2)
    
    def __repr__(self) -> str:
        """Representation."""
        return f"Config({self._config})"


def create_default_config() -> Config:
    """
    Create a default configuration for CPU allocation model.
    
    Returns:
        Default configuration
    """
    default_config = {
        'experiment_name': 'cpu_allocation_experiment',
        'task_type': 'regression',
        'device': 'cpu',
        'output_dir': 'experiments',
        'log_level': 'INFO',
        
        'data': {
            'source': 'synthetic',
            'n_samples': 10000,
            'n_features': 10,
            'sequence_length': None,
            'noise_level': 0.1,
            'normalize': True,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'batch_size': 32,
            'num_workers': 0,
        },
        
        'model': {
            'name': 'cpu_allocation_mlp',
            'input_dim': 10,
            'hidden_dims': [128, 64, 32],
            'output_dim': 1,
            'dropout_rate': 0.1,
            'activation': 'relu',
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'allocation_type': 'percentage',
            'max_cpu_cores': 16,
        },
        
        'training': {
            'epochs': 100,
            'save_every': 10,
            'validate_every': 1,
            'early_stopping_patience': 10,
            'gradient_clip_val': None,
            'scheduler': {
                'type': 'StepLR',
                'params': {
                    'step_size': 30,
                    'gamma': 0.1
                }
            }
        },
        
        'evaluation': {
            'cv_epochs': 20,
            'benchmark_runs': 100,
        }
    }
    
    return Config(default_config)


def load_config_with_overrides(base_config_path: Union[str, Path],
                             overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration with optional overrides.
    
    Args:
        base_config_path: Path to base configuration file
        overrides: Dictionary of override values
        
    Returns:
        Configuration with overrides applied
    """
    config = Config.from_yaml(base_config_path)
    
    if overrides:
        config.update(overrides)
    
    return config


# Configuration schema for validation
CONFIG_SCHEMA = {
    'experiment_name': {'type': str, 'required': True},
    'task_type': {'type': str, 'required': True},
    'device': {'type': str, 'required': False},
    'output_dir': {'type': str, 'required': False},
    'log_level': {'type': str, 'required': False},
    
    'data': {
        'source': {'type': str, 'required': True},
        'batch_size': {'type': int, 'required': True},
        'train_ratio': {'type': float, 'required': True},
        'val_ratio': {'type': float, 'required': True},
        'test_ratio': {'type': float, 'required': True},
    },
    
    'model': {
        'name': {'type': str, 'required': True},
        'learning_rate': {'type': float, 'required': True},
    },
    
    'training': {
        'epochs': {'type': int, 'required': True},
    }
}

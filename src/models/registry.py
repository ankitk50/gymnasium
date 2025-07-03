"""
Model registry for managing different model implementations.
"""

from typing import Dict, Type, Any
from ..core.base_model import BaseModel


class ModelRegistry:
    """
    Registry for managing different model implementations.
    """
    
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class.
        
        Args:
            name: Model name
            model_class: Model class to register
        """
        cls._models[name] = model_class
    
    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        """
        Get a model class by name.
        
        Args:
            name: Model name
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model is not found
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(cls._models.keys())}")
        
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            name: Model name
            
        Returns:
            Model information dictionary
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found")
        
        model_class = cls._models[name]
        
        return {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'docstring': model_class.__doc__
        }


# Register CPU allocation models
def register_cpu_allocation_models():
    """Register all CPU allocation models."""
    from .cpu_allocation import (
        CPUAllocationMLP,
        CPUAllocationLSTM,
        CPUAllocationTransformer
    )
    
    ModelRegistry.register('cpu_allocation_mlp', CPUAllocationMLP)
    ModelRegistry.register('cpu_allocation_lstm', CPUAllocationLSTM)
    ModelRegistry.register('cpu_allocation_transformer', CPUAllocationTransformer)


# Register models on import
register_cpu_allocation_models()


# Example of registering custom models
"""
To register a custom model:

from src.models.registry import ModelRegistry
from src.core.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your implementation
    
    def forward(self, x):
        # Your forward pass
        pass
    
    def get_loss_function(self):
        # Return loss function
        pass
    
    def get_optimizer(self, parameters):
        # Return optimizer
        pass

# Register the model
ModelRegistry.register('custom_model', CustomModel)
"""

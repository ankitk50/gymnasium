"""
Base model interface for the training pipeline framework.
All models should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from pathlib import Path


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models in the framework.
    
    This class defines the interface that all models must implement
    to be compatible with the training pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model_name = config.get('name', self.__class__.__name__)
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Model metadata
        self.input_shape = config.get('input_shape')
        self.output_shape = config.get('output_shape')
        self.num_parameters = 0
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_loss_function(self) -> nn.Module:
        """
        Get the loss function for this model.
        
        Returns:
            Loss function
        """
        pass
    
    @abstractmethod
    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        """
        Get the optimizer for this model.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Optimizer instance
        """
        pass
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Get the learning rate scheduler (optional).
        
        Args:
            optimizer: Optimizer instance
            
        Returns:
            Learning rate scheduler or None
        """
        scheduler_config = self.config.get('scheduler')
        if not scheduler_config:
            return None
            
        scheduler_type = scheduler_config.get('type', 'StepLR')
        scheduler_params = scheduler_config.get('params', {})
        
        if scheduler_type == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
        elif scheduler_type == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.num_parameters
    
    def save_checkpoint(self, path: Union[str, Path], optimizer: Optional[torch.optim.Optimizer] = None, 
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                       epoch: int = 0, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            metrics: Training metrics (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'epoch': epoch,
            'num_parameters': self.count_parameters(),
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if metrics:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path], 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'num_parameters': checkpoint.get('num_parameters', 0)
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary information.
        
        Returns:
            Model summary dictionary
        """
        return {
            'name': self.model_name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'num_parameters': self.count_parameters(),
            'device': str(self.device),
            'config': self.config
        }

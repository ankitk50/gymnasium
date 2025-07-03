"""
CPU allocation models for task scheduling optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np

from ..core.base_model import BaseModel


class CPUAllocationMLP(BaseModel):
    """
    Multi-layer perceptron for CPU allocation prediction.
    Predicts optimal CPU allocation given task characteristics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CPU allocation MLP.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Model architecture parameters
        self.input_dim = config.get('input_dim', 10)  # Task features
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32])
        self.output_dim = config.get('output_dim', 1)  # CPU allocation (0-1 or actual cores)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.activation = config.get('activation', 'relu')
        
        # Task-specific parameters
        self.max_cpu_cores = config.get('max_cpu_cores', 16)
        self.allocation_type = config.get('allocation_type', 'percentage')  # 'percentage' or 'cores'
        
        # Build the network
        self._build_network()
        
        # Loss and optimization
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-5)
        
    def _build_network(self):
        """Build the neural network layers."""
        layers = []
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        # Output activation based on allocation type
        if self.allocation_type == 'percentage':
            layers.append(nn.Sigmoid())  # Output between 0 and 1
        elif self.allocation_type == 'cores':
            layers.extend([
                nn.ReLU(),  # Ensure positive values
                nn.Linear(self.output_dim, self.max_cpu_cores + 1)  # Discrete cores
            ])
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self):
        """Get activation function based on config."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()  # Default
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Features: [task_priority, memory_requirement, io_intensity, 
                         computation_complexity, deadline_urgency, dependency_count,
                         historical_runtime, resource_contention, user_priority, workload_type]
        
        Returns:
            CPU allocation prediction
        """
        return self.network(x)
    
    def get_loss_function(self) -> nn.Module:
        """Get the loss function for CPU allocation."""
        if self.allocation_type == 'percentage':
            return nn.MSELoss()  # Regression for percentage allocation
        elif self.allocation_type == 'cores':
            return nn.CrossEntropyLoss()  # Classification for discrete cores
        else:
            return nn.MSELoss()
    
    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Get the optimizer for training."""
        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


class CPUAllocationLSTM(BaseModel):
    """
    LSTM-based model for CPU allocation with temporal dependencies.
    Considers task history and temporal patterns for allocation decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CPU allocation LSTM.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Model architecture parameters
        self.input_dim = config.get('input_dim', 10)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.output_dim = config.get('output_dim', 1)
        self.sequence_length = config.get('sequence_length', 10)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Task-specific parameters
        self.max_cpu_cores = config.get('max_cpu_cores', 16)
        self.allocation_type = config.get('allocation_type', 'percentage')
        
        # Build the network
        self._build_network()
        
        # Loss and optimization
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-5)
    
    def _build_network(self):
        """Build the LSTM network."""
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
        # Output activation
        if self.allocation_type == 'percentage':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            CPU allocation prediction
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention (use last hidden state as query)
        last_hidden = lstm_out[:, -1:, :]  # Shape: (batch_size, 1, hidden_dim)
        attended_out, _ = self.attention(last_hidden, lstm_out, lstm_out)
        
        # Use attended output
        output = attended_out.squeeze(1)  # Shape: (batch_size, hidden_dim)
        
        # Final prediction
        output = self.fc_layers(output)
        output = self.output_activation(output)
        
        return output
    
    def get_loss_function(self) -> nn.Module:
        """Get the loss function for CPU allocation."""
        if self.allocation_type == 'percentage':
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()
    
    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Get the optimizer for training."""
        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


class CPUAllocationTransformer(BaseModel):
    """
    Transformer-based model for CPU allocation with attention mechanisms.
    Captures complex dependencies between tasks and system state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CPU allocation Transformer.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        # Model architecture parameters
        self.input_dim = config.get('input_dim', 10)
        self.model_dim = config.get('model_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.ff_dim = config.get('ff_dim', 512)
        self.output_dim = config.get('output_dim', 1)
        self.max_seq_length = config.get('max_seq_length', 100)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Task-specific parameters
        self.max_cpu_cores = config.get('max_cpu_cores', 16)
        self.allocation_type = config.get('allocation_type', 'percentage')
        
        # Build the network
        self._build_network()
        
        # Loss and optimization
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.weight_decay = config.get('weight_decay', 1e-5)
    
    def _build_network(self):
        """Build the Transformer network."""
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.model_dim)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.model_dim // 2, self.output_dim)
        )
        
        # Output activation
        if self.allocation_type == 'percentage':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.ReLU()
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create positional encoding for sequences."""
        pe = torch.zeros(self.max_seq_length, self.model_dim)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.model_dim, 2).float() *
                           -(np.log(10000.0) / self.model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Shape: (1, max_seq_length, model_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
        
        Returns:
            CPU allocation prediction
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_length, model_dim)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_length, :].to(x.device)
        x = x + pos_encoding
        
        # Transformer encoding
        encoded = self.transformer(x)  # (batch_size, seq_length, model_dim)
        
        # Global average pooling or use last token
        pooled = encoded.mean(dim=1)  # (batch_size, model_dim)
        
        # Output projection
        output = self.output_projection(pooled)
        output = self.output_activation(output)
        
        return output
    
    def get_loss_function(self) -> nn.Module:
        """Get the loss function for CPU allocation."""
        if self.allocation_type == 'percentage':
            # Custom loss that considers efficiency
            return CPUAllocationLoss(allocation_type=self.allocation_type)
        else:
            return nn.CrossEntropyLoss()
    
    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        """Get the optimizer for training."""
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


class CPUAllocationLoss(nn.Module):
    """
    Custom loss function for CPU allocation that considers both
    prediction accuracy and resource efficiency.
    """
    
    def __init__(self, allocation_type: str = 'percentage', 
                 efficiency_weight: float = 0.1):
        """
        Initialize the custom loss function.
        
        Args:
            allocation_type: Type of allocation prediction
            efficiency_weight: Weight for efficiency term
        """
        super().__init__()
        self.allocation_type = allocation_type
        self.efficiency_weight = efficiency_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the custom CPU allocation loss.
        
        Args:
            predictions: Predicted CPU allocations
            targets: Target CPU allocations
            
        Returns:
            Combined loss value
        """
        # Base prediction loss
        prediction_loss = self.mse_loss(predictions, targets)
        
        # Efficiency penalty (penalize over-allocation)
        over_allocation = torch.clamp(predictions - targets, min=0)
        efficiency_loss = torch.mean(over_allocation ** 2)
        
        # Under-allocation penalty (penalize under-allocation more heavily)
        under_allocation = torch.clamp(targets - predictions, min=0)
        under_allocation_loss = torch.mean(under_allocation ** 2) * 2  # Higher penalty
        
        # Combined loss
        total_loss = (prediction_loss + 
                     self.efficiency_weight * efficiency_loss +
                     self.efficiency_weight * under_allocation_loss)
        
        return total_loss


# Utility functions for CPU allocation
def generate_cpu_allocation_features(task_info: Dict[str, Any]) -> torch.Tensor:
    """
    Generate feature vector from task information.
    
    Args:
        task_info: Dictionary containing task information
        
    Returns:
        Feature tensor
    """
    features = [
        task_info.get('priority', 0.5),              # Task priority (0-1)
        task_info.get('memory_requirement', 0.5),    # Memory requirement (0-1, normalized)
        task_info.get('io_intensity', 0.3),          # I/O intensity (0-1)
        task_info.get('computation_complexity', 0.5), # Computational complexity (0-1)
        task_info.get('deadline_urgency', 0.5),      # Deadline urgency (0-1)
        task_info.get('dependency_count', 0.0),      # Number of dependencies (normalized)
        task_info.get('historical_runtime', 0.5),    # Historical runtime (normalized)
        task_info.get('resource_contention', 0.3),   # Current resource contention (0-1)
        task_info.get('user_priority', 0.5),         # User priority (0-1)
        task_info.get('workload_type', 0.5),         # Workload type encoding (0-1)
    ]
    
    return torch.tensor(features, dtype=torch.float32)


def calculate_allocation_efficiency(predictions: torch.Tensor, 
                                  actual_usage: torch.Tensor) -> Dict[str, float]:
    """
    Calculate efficiency metrics for CPU allocation.
    
    Args:
        predictions: Predicted CPU allocations
        actual_usage: Actual CPU usage
        
    Returns:
        Efficiency metrics dictionary
    """
    predictions_np = predictions.detach().cpu().numpy()
    actual_np = actual_usage.detach().cpu().numpy()
    
    # Calculate metrics
    over_allocation = np.maximum(predictions_np - actual_np, 0)
    under_allocation = np.maximum(actual_np - predictions_np, 0)
    
    efficiency_metrics = {
        'mean_over_allocation': np.mean(over_allocation),
        'mean_under_allocation': np.mean(under_allocation),
        'allocation_accuracy': 1.0 - np.mean(np.abs(predictions_np - actual_np)),
        'resource_utilization': np.mean(actual_np / np.maximum(predictions_np, 1e-8)),
        'waste_ratio': np.mean(over_allocation / np.maximum(predictions_np, 1e-8))
    }
    
    return efficiency_metrics

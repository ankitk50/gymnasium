"""
Custom metrics and evaluation utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import time


class MetricsTracker:
    """
    Track and compute metrics during training and evaluation.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self) -> Dict[str, float]:
        """
        Reset all metrics.
        
        Returns:
            Empty metrics dictionary
        """
        self.metrics = defaultdict(list)
        self.running_metrics = {}
        return {}
    
    def update(self, batch_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update metrics with batch results.
        
        Args:
            batch_metrics: Metrics from current batch
            
        Returns:
            Current running averages
        """
        for metric, value in batch_metrics.items():
            self.metrics[metric].append(value)
        
        # Compute running averages
        self.running_metrics = {
            metric: np.mean(values) for metric, values in self.metrics.items()
        }
        
        return self.running_metrics.copy()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics.
        
        Returns:
            Current metrics dictionary
        """
        return self.running_metrics.copy()
    
    def get_metric_history(self, metric: str) -> List[float]:
        """
        Get history of a specific metric.
        
        Args:
            metric: Metric name
            
        Returns:
            List of metric values
        """
        return self.metrics.get(metric, [])


class CPUAllocationMetrics:
    """
    Specialized metrics for CPU allocation tasks.
    """
    
    @staticmethod
    def allocation_accuracy(predictions: torch.Tensor, targets: torch.Tensor,
                          tolerance: float = 0.05) -> float:
        """
        Compute allocation accuracy within tolerance.
        
        Args:
            predictions: Predicted allocations
            targets: Target allocations
            tolerance: Acceptable tolerance for accuracy
            
        Returns:
            Accuracy score
        """
        diff = torch.abs(predictions - targets)
        accurate = (diff <= tolerance).float()
        return accurate.mean().item()
    
    @staticmethod
    def resource_efficiency(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute resource efficiency metrics.
        
        Args:
            predictions: Predicted allocations
            targets: Target allocations
            
        Returns:
            Efficiency metrics dictionary
        """
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Over-allocation (waste)
        over_allocation = np.maximum(predictions_np - targets_np, 0)
        waste_ratio = np.mean(over_allocation) / np.mean(predictions_np)
        
        # Under-allocation (missed opportunities)
        under_allocation = np.maximum(targets_np - predictions_np, 0)
        under_allocation_ratio = np.mean(under_allocation) / np.mean(targets_np)
        
        # Resource utilization
        utilization = np.mean(targets_np / np.maximum(predictions_np, 1e-8))
        
        return {
            'waste_ratio': waste_ratio,
            'under_allocation_ratio': under_allocation_ratio,
            'resource_utilization': utilization,
            'efficiency_score': 1.0 - waste_ratio - under_allocation_ratio
        }
    
    @staticmethod
    def allocation_stability(predictions: torch.Tensor) -> float:
        """
        Compute allocation stability (low variance indicates stable allocations).
        
        Args:
            predictions: Predicted allocations
            
        Returns:
            Stability score (1 - normalized_variance)
        """
        variance = torch.var(predictions).item()
        mean_allocation = torch.mean(predictions).item()
        
        if mean_allocation == 0:
            return 1.0
        
        normalized_variance = variance / (mean_allocation ** 2)
        stability = 1.0 / (1.0 + normalized_variance)
        
        return stability
    
    @staticmethod
    def fairness_score(predictions: torch.Tensor, task_priorities: torch.Tensor) -> float:
        """
        Compute fairness score based on task priorities.
        
        Args:
            predictions: Predicted allocations
            task_priorities: Task priority scores
            
        Returns:
            Fairness score
        """
        # Compute allocation per unit priority
        allocation_per_priority = predictions / torch.clamp(task_priorities, min=1e-8)
        
        # Fairness is measured as 1 - coefficient of variation
        mean_ratio = torch.mean(allocation_per_priority)
        std_ratio = torch.std(allocation_per_priority)
        
        if mean_ratio == 0:
            return 1.0
        
        coefficient_of_variation = std_ratio / mean_ratio
        fairness = 1.0 / (1.0 + coefficient_of_variation.item())
        
        return fairness


class PerformanceTimer:
    """
    Timer for measuring performance metrics.
    """
    
    def __init__(self):
        """Initialize performance timer."""
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str) -> None:
        """
        Start timing an operation.
        
        Args:
            name: Operation name
        """
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """
        Stop timing an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.start_times[name]
        
        if name not in self.times:
            self.times[name] = []
        
        self.times[name].append(elapsed)
        del self.start_times[name]
        
        return elapsed
    
    def get_average_time(self, name: str) -> float:
        """
        Get average time for an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Average time in seconds
        """
        if name not in self.times or not self.times[name]:
            return 0.0
        
        return np.mean(self.times[name])
    
    def get_total_time(self, name: str) -> float:
        """
        Get total time for an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Total time in seconds
        """
        if name not in self.times:
            return 0.0
        
        return sum(self.times[name])
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Statistics dictionary
        """
        if name not in self.times or not self.times[name]:
            return {}
        
        times = self.times[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': min(times),
            'max': max(times)
        }
    
    def context_timer(self, name: str):
        """
        Context manager for timing operations.
        
        Args:
            name: Operation name
            
        Returns:
            Context manager
        """
        return TimerContext(self, name)


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, timer: PerformanceTimer, name: str):
        """
        Initialize timer context.
        
        Args:
            timer: Performance timer instance
            name: Operation name
        """
        self.timer = timer
        self.name = name
    
    def __enter__(self):
        """Enter context."""
        self.timer.start(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.timer.stop(self.name)


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.Inf
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
            
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def compute_model_complexity(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Compute model complexity metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Complexity metrics dictionary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'parameter_ratio': trainable_params / total_params if total_params > 0 else 0
    }


def calculate_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """
    Estimate FLOPs (Floating Point Operations) for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        
    Returns:
        Estimated FLOPs
    """
    def flop_count_hook(module, input, output):
        """Hook to count FLOPs for different layer types."""
        flops = 0
        
        if isinstance(module, torch.nn.Linear):
            # For linear layers: input_features * output_features
            flops = module.in_features * module.out_features
            if module.bias is not None:
                flops += module.out_features
        
        elif isinstance(module, torch.nn.Conv1d):
            # For 1D convolution
            output_elements = output.numel()
            kernel_flops = module.kernel_size[0] * module.in_channels
            flops = output_elements * kernel_flops
        
        elif isinstance(module, torch.nn.Conv2d):
            # For 2D convolution
            output_elements = output.numel()
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            flops = output_elements * kernel_flops
        
        module.__flops__ += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        module.__flops__ = 0
        hooks.append(module.register_forward_hook(flop_count_hook))
    
    # Forward pass with dummy input
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        model(dummy_input)
    
    # Sum up FLOPs
    total_flops = sum(module.__flops__ for module in model.modules())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops

"""
Output directory configuration helper.
This module provides utilities to ensure consistent output directory management
across all components of the ML/DL training pipeline framework.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import os

class OutputManager:
    """
    Centralized output directory management for the training pipeline.
    Ensures all outputs (models, logs, visualizations, etc.) are saved
    in a unified structure within the results directory.
    """
    
    def __init__(self, base_dir: str = "results", experiment_name: Optional[str] = None):
        """
        Initialize the output manager.
        
        Args:
            base_dir: Base directory for all outputs (default: 'results')
            experiment_name: Name of the current experiment
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        
        # Create base directory structure
        self._create_base_structure()
    
    def _create_base_structure(self):
        """Create the basic directory structure."""
        directories = [
            self.base_dir,
            self.base_dir / "experiments",
            self.base_dir / "inference", 
            self.base_dir / "validation",
            self.base_dir / "deployment",
            self.base_dir / "logs",
            self.base_dir / "visualizations",
            self.base_dir / "models",
            self.base_dir / "metrics",
            self.base_dir / "exports",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_experiment_dir(self, experiment_name: Optional[str] = None) -> Path:
        """
        Get the directory for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment (uses self.experiment_name if None)
            
        Returns:
            Path to the experiment directory
        """
        name = experiment_name or self.experiment_name or "default_experiment"
        exp_dir = self.base_dir / "experiments" / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def get_model_dir(self, experiment_name: Optional[str] = None) -> Path:
        """Get the models directory for an experiment."""
        exp_dir = self.get_experiment_dir(experiment_name)
        model_dir = exp_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def get_log_dir(self, experiment_name: Optional[str] = None) -> Path:
        """Get the logs directory for an experiment."""
        exp_dir = self.get_experiment_dir(experiment_name)
        log_dir = exp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def get_visualization_dir(self, experiment_name: Optional[str] = None) -> Path:
        """Get the visualizations directory for an experiment."""
        exp_dir = self.get_experiment_dir(experiment_name)
        viz_dir = exp_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        return viz_dir
    
    def get_metrics_dir(self, experiment_name: Optional[str] = None) -> Path:
        """Get the metrics directory for an experiment."""
        exp_dir = self.get_experiment_dir(experiment_name)
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        return metrics_dir
    
    def get_inference_dir(self) -> Path:
        """Get the inference directory."""
        inf_dir = self.base_dir / "inference"
        inf_dir.mkdir(parents=True, exist_ok=True)
        return inf_dir
    
    def get_validation_dir(self) -> Path:
        """Get the validation directory."""
        val_dir = self.base_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        return val_dir
    
    def get_deployment_dir(self) -> Path:
        """Get the deployment directory."""
        dep_dir = self.base_dir / "deployment"
        dep_dir.mkdir(parents=True, exist_ok=True)
        return dep_dir
    
    def get_export_dir(self) -> Path:
        """Get the exports directory."""
        exp_dir = self.base_dir / "exports"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def get_demo_dir(self) -> Path:
        """Get the demo directory."""
        demo_dir = self.base_dir / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        return demo_dir
    
    def get_paths_config(self, experiment_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get a configuration dictionary with all relevant paths.
        Useful for passing to components that need multiple paths.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary with path configurations
        """
        return {
            'experiment_dir': str(self.get_experiment_dir(experiment_name)),
            'model_dir': str(self.get_model_dir(experiment_name)),
            'log_dir': str(self.get_log_dir(experiment_name)),
            'visualization_dir': str(self.get_visualization_dir(experiment_name)),
            'metrics_dir': str(self.get_metrics_dir(experiment_name)),
            'inference_dir': str(self.get_inference_dir()),
            'validation_dir': str(self.get_validation_dir()),
            'deployment_dir': str(self.get_deployment_dir()),
            'export_dir': str(self.get_export_dir()),
            'demo_dir': str(self.get_demo_dir()),
        }
    
    def setup_tensorboard_logging(self, experiment_name: Optional[str] = None) -> str:
        """
        Setup TensorBoard logging directory.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Path to TensorBoard log directory
        """
        exp_dir = self.get_experiment_dir(experiment_name)
        tb_dir = exp_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        return str(tb_dir)


# Global instance for easy access
default_output_manager = OutputManager()

def get_output_manager(base_dir: str = "results", experiment_name: Optional[str] = None) -> OutputManager:
    """
    Get an output manager instance.
    
    Args:
        base_dir: Base directory for outputs
        experiment_name: Experiment name
        
    Returns:
        OutputManager instance
    """
    return OutputManager(base_dir, experiment_name)

def ensure_results_structure():
    """Ensure the results directory structure exists."""
    default_output_manager._create_base_structure()

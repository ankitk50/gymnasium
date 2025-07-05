"""
Initialize the ML/DL training pipeline package.
"""

__version__ = "1.0.0"
__author__ = "ML Pipeline Framework"
__description__ = "A generic reusable training pipeline framework for ML/DL models"

# Core imports
from .core.pipeline import TrainingPipeline
from .core.base_model import BaseModel
from .core.trainer import Trainer
from .core.evaluator import Evaluator
from .core.inference import InferenceEngine
from .core.validation import ModelValidator
from .core.deployment import ModelDeploymentManager, InferenceServer

# Utility imports
from .utils.config import Config, create_default_config
from .utils.logging import setup_logging
from .utils.metrics import MetricsTracker, CPUAllocationMetrics

# Model imports
from .models.registry import ModelRegistry
from .models.cpu_allocation import (
    CPUAllocationMLP, 
    CPUAllocationLSTM, 
    CPUAllocationTransformer
)

# Data imports
from .data.data_loader import CPUAllocationDataset, create_data_loaders
from .data.preprocessor import DataPreprocessor

# Visualization imports
from .visualization.training_viz import TrainingVisualizer
from .visualization.evaluation_viz import EvaluationVisualizer
from .visualization.inference_viz import InferenceVisualizer

__all__ = [
    # Core classes
    'TrainingPipeline',
    'BaseModel', 
    'Trainer',
    'Evaluator',
    'InferenceEngine',
    'ModelValidator', 
    'ModelDeploymentManager',
    'InferenceServer',
    
    # Utilities
    'Config',
    'create_default_config',
    'setup_logging',
    'MetricsTracker',
    'CPUAllocationMetrics',
    
    # Models
    'ModelRegistry',
    'CPUAllocationMLP',
    'CPUAllocationLSTM', 
    'CPUAllocationTransformer',
    
    # Data
    'CPUAllocationDataset',
    'create_data_loaders',
    'DataPreprocessor',
    
    # Visualization
    'TrainingVisualizer',
    'EvaluationVisualizer',
    'InferenceVisualizer',
]

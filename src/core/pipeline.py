"""
Main training pipeline that orchestrates the entire ML/DL workflow.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader

# Wandb integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .base_model import BaseModel
from .trainer import Trainer
from .evaluator import Evaluator
from ..data.data_loader import create_data_loaders
from ..models.registry import ModelRegistry
from ..utils.config import Config
from ..utils.logging import setup_logging
from ..visualization.training_viz import TrainingVisualizer


class TrainingPipeline:
    """
    Main training pipeline that orchestrates the entire ML/DL workflow.
    
    This class provides a high-level interface for:
    - Loading and preprocessing data
    - Initializing models
    - Training models
    - Evaluating models
    - Saving results and visualizations
    """
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path]] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 experiment_name: Optional[str] = None,
                 use_wandb: bool = False,
                 wandb_project: Optional[str] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file
            config_dict: Configuration dictionary (alternative to config_path)
            experiment_name: Name for this experiment
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: Wandb project name
        """
        # Load configuration
        if config_path:
            self.config = Config.from_yaml(config_path)
        elif config_dict:
            self.config = Config(config_dict)
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Set experiment name
        self.experiment_name = experiment_name or self.config.get('experiment_name', 'default_experiment')
        
        # Setup wandb if requested
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "ml-pipeline-experiments",
                name=self.experiment_name,
                config=self.config.to_dict()
            )
        
        # Setup output directory
        self.output_dir = Path(self.config.get('output_dir', 'results')) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.get('log_level', 'INFO'),
            log_file=self.output_dir / 'training.log'
        )
        
        # Save configuration
        self._save_config()
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.logger.info(f"Initialized training pipeline for experiment: {self.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_data(self) -> None:
        """Setup data loaders based on configuration."""
        self.logger.info("Setting up data loaders...")
        
        data_config = self.config.get('data', {})
        
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_config=data_config,
            task_type=self.config.get('task_type', 'regression')
        )
        
        self.logger.info(f"Train loader: {len(self.train_loader)} batches")
        if self.val_loader:
            self.logger.info(f"Validation loader: {len(self.val_loader)} batches")
        if self.test_loader:
            self.logger.info(f"Test loader: {len(self.test_loader)} batches")
    
    def setup_model(self) -> None:
        """Setup model based on configuration."""
        self.logger.info("Setting up model...")
        
        model_config = self.config.get('model', {})
        model_name = model_config.get('name')
        
        if not model_name:
            raise ValueError("Model name not specified in configuration")
        
        # Get model from registry
        model_class = ModelRegistry.get_model(model_name)
        
        # Add global config to model config
        full_model_config = {
            **model_config,
            'task_type': self.config.get('task_type'),
            'device': self.config.get('device', 'cpu'),
            'output_dir': str(self.output_dir)
        }
        
        self.model = model_class(full_model_config)
        
        self.logger.info(f"Initialized model: {model_name}")
        self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        # Print model summary
        summary = self.model.get_model_summary()
        for key, value in summary.items():
            if key != 'config':  # Skip detailed config in log
                self.logger.info(f"  {key}: {value}")
    
    def train(self) -> BaseModel:
        """
        Train the model.
        
        Returns:
            Trained model
        """
        if self.model is None:
            self.setup_model()
        
        if self.train_loader is None:
            self.setup_data()
        
        self.logger.info("Starting model training...")
        
        # Setup trainer
        training_config = {
            **self.config.get('training', {}),
            'task_type': self.config.get('task_type'),
            'output_dir': str(self.output_dir)
        }
        
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=training_config,
            logger=self.logger
        )
        
        # Train model
        trained_model = self.trainer.train()
        
        self.logger.info("Training completed successfully!")
        
        return trained_model
    
    def evaluate(self, model: Optional[BaseModel] = None, 
                test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            model: Model to evaluate (uses trained model if None)
            test_loader: Test data loader (uses default if None)
            
        Returns:
            Evaluation results
        """
        # Use provided model or the trained model
        eval_model = model or self.model
        if eval_model is None:
            raise ValueError("No model available for evaluation. Train a model first or provide one.")
        
        # Use provided test loader or the default one
        eval_test_loader = test_loader or self.test_loader
        if eval_test_loader is None:
            raise ValueError("No test data available for evaluation.")
        
        self.logger.info("Starting model evaluation...")
        
        # Setup evaluator
        evaluation_config = {
            **self.config.get('evaluation', {}),
            'task_type': self.config.get('task_type'),
            'output_dir': str(self.output_dir)
        }
        
        self.evaluator = Evaluator(eval_model, evaluation_config)
        
        # Evaluate model
        results = self.evaluator.evaluate(eval_test_loader)
        
        # Log key metrics
        metrics = results['metrics']
        self.logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric}: {value:.4f}")
        
        self.logger.info("Evaluation completed successfully!")
        
        return results
    
    def cross_validate(self, k_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            k_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        if self.model is None:
            self.setup_model()
        
        if self.train_loader is None:
            self.setup_data()
        
        self.logger.info(f"Starting {k_folds}-fold cross-validation...")
        
        # Setup evaluator for cross-validation
        evaluation_config = {
            **self.config.get('evaluation', {}),
            **self.config.get('training', {}),
            'task_type': self.config.get('task_type'),
            'output_dir': str(self.output_dir)
        }
        
        evaluator = Evaluator(self.model, evaluation_config)
        
        # Perform cross-validation
        cv_results = evaluator.cross_validate(self.train_loader, k_folds)
        
        # Log results
        cv_metrics = cv_results['cv_metrics']
        self.logger.info("Cross-Validation Results:")
        for metric, value in cv_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return cv_results
    
    def hyperparameter_search(self, search_space: Dict[str, Any], 
                            n_trials: int = 50) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters and results
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter search. Install with: pip install optuna")
        
        self.logger.info(f"Starting hyperparameter search with {n_trials} trials...")
        
        def objective(trial):
            # Sample hyperparameters
            trial_config = self.config.copy()
            
            for param_path, param_config in search_space.items():
                param_type = param_config['type']
                param_range = param_config['range']
                
                if param_type == 'float':
                    value = trial.suggest_float(param_path, *param_range)
                elif param_type == 'int':
                    value = trial.suggest_int(param_path, *param_range)
                elif param_type == 'categorical':
                    value = trial.suggest_categorical(param_path, param_range)
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
                
                # Set parameter in config (supports nested paths like 'model.learning_rate')
                keys = param_path.split('.')
                current = trial_config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
            
            # Create temporary pipeline with trial configuration
            trial_pipeline = TrainingPipeline(
                config_dict=trial_config,
                experiment_name=f"{self.experiment_name}_trial_{trial.number}"
            )
            
            # Train and evaluate
            trial_pipeline.setup_data()
            trained_model = trial_pipeline.train()
            results = trial_pipeline.evaluate(trained_model)
            
            # Return objective metric (minimize validation loss)
            return results['metrics'].get('test_loss', float('inf'))
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Log best results
        self.logger.info("Hyperparameter search completed!")
        self.logger.info(f"Best value: {study.best_value:.4f}")
        self.logger.info("Best parameters:")
        for param, value in study.best_params.items():
            self.logger.info(f"  {param}: {value}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def inference(self, data_loader: DataLoader, 
                 model: Optional[BaseModel] = None) -> Dict[str, Any]:
        """
        Run inference on new data.
        
        Args:
            data_loader: Data loader for inference
            model: Model to use (uses trained model if None)
            
        Returns:
            Inference results
        """
        # Use provided model or the trained model
        inference_model = model or self.model
        if inference_model is None:
            raise ValueError("No model available for inference. Train a model first or provide one.")
        
        self.logger.info("Running model inference...")
        
        inference_model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(inference_model.device)
                output = inference_model(data)
                
                if self.config.get('task_type') == 'classification':
                    pred = output.argmax(dim=1)
                else:
                    pred = output
                
                predictions.extend(pred.cpu().numpy())
        
        self.logger.info(f"Inference completed on {len(predictions)} samples")
        
        return {
            'predictions': predictions,
            'num_samples': len(predictions)
        }
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> BaseModel:
        """
        Load a model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model
        """
        if self.model is None:
            self.setup_model()
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_info = self.model.load_checkpoint(checkpoint_path)
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
        
        return self.model
    
    def _save_config(self) -> None:
        """Save the configuration to the output directory."""
        config_path = self.output_dir / 'config.yaml'
        self.config.save_yaml(config_path)
        self.logger.info(f"Configuration saved to {config_path}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the experiment.
        
        Returns:
            Experiment summary dictionary
        """
        summary = {
            'experiment_name': self.experiment_name,
            'output_dir': str(self.output_dir),
            'config': self.config.to_dict(),
        }
        
        if self.model:
            summary['model_summary'] = self.model.get_model_summary()
        
        if self.trainer:
            summary['training_history'] = self.trainer.get_training_history()
        
        return summary

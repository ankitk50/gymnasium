"""
Core evaluator module for model evaluation and testing.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

from pathlib import Path
import json

from .base_model import BaseModel
from ..visualization.evaluation_viz import EvaluationVisualizer


class Evaluator:
    """
    Model evaluator for comprehensive performance assessment.
    """
    
    def __init__(self, 
                 model: BaseModel, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            config: Evaluation configuration
        """
        self.model = model
        self.config = config or {}
        self.device = model.device
        self.task_type = self.config.get('task_type', 'regression')
        
        self.visualizer = EvaluationVisualizer(
            self.config.get('output_dir', 'experiments')
        )
        
        # Evaluation results storage
        self.results = {}
        
    def evaluate(self, test_loader: DataLoader, 
                save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            test_loader: Test data loader
            save_results: Whether to save results to disk
            
        Returns:
            Evaluation results dictionary
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_losses = []
        
        criterion = self.model.get_loss_function()
        
        # Collect predictions and targets
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                if self.task_type == 'classification':
                    predictions = output.argmax(dim=1)
                else:
                    predictions = output
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_losses.append(loss.item())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        avg_loss = np.mean(all_losses)
        
        # Compute metrics based on task type
        if self.task_type == 'classification':
            metrics = self._compute_classification_metrics(predictions, targets)
        else:
            metrics = self._compute_regression_metrics(predictions, targets)
        
        metrics['test_loss'] = avg_loss
        
        # Store results
        self.results = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'losses': all_losses
        }
        
        # Generate visualizations
        self._generate_evaluation_plots(predictions, targets, metrics)
        
        # Save results
        if save_results:
            self._save_results()
        
        return self.results
    
    def _compute_classification_metrics(self, predictions: np.ndarray, 
                                      targets: np.ndarray) -> Dict[str, Any]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Classification metrics dictionary
        """
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(targets, predictions, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def _compute_regression_metrics(self, predictions: np.ndarray, 
                                  targets: np.ndarray) -> Dict[str, Any]:
        """
        Compute regression metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Regression metrics dictionary
        """
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # Additional regression metrics
        mape = np.mean(np.abs((targets - predictions) / np.clip(targets, 1e-8, None))) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape
        }
    
    def _generate_evaluation_plots(self, predictions: np.ndarray, 
                                 targets: np.ndarray, 
                                 metrics: Dict[str, Any]) -> None:
        """
        Generate evaluation visualizations.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            metrics: Computed metrics
        """
        try:
            # Convert tensors to numpy arrays for visualization
            if isinstance(targets, torch.Tensor):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = np.asarray(targets)
                
            if isinstance(predictions, torch.Tensor):
                predictions_np = predictions.cpu().numpy()
            else:
                predictions_np = np.asarray(predictions)
                
            if self.task_type == 'classification':
                self.visualizer.plot_confusion_matrix(
                    metrics['confusion_matrix'],
                    title='Confusion Matrix'
                )
                self.visualizer.plot_classification_report(
                    metrics['classification_report']
                )
            else:
                self.visualizer.plot_regression_results(
                    targets_np, predictions_np, metrics
                )
                self.visualizer.plot_residuals(targets_np, predictions_np)
        except Exception as e:
            print(f"Warning: Failed to generate evaluation plots: {e}")
            # Continue without plots rather than failing the entire evaluation
    
    def cross_validate(self, train_loader: DataLoader, 
                      k_folds: int = 5) -> Dict[str, Any]:
        """
        Perform k-fold cross validation.
        
        Args:
            train_loader: Training data loader
            k_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        # Get all data from loader
        all_data = []
        all_targets = []
        
        for data, target in train_loader:
            all_data.append(data)
            all_targets.append(target)
        
        all_data = torch.cat(all_data, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Convert to numpy for sklearn KFold
        all_data_np = all_data.numpy()
        all_targets_np = all_targets.numpy()
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_data_np)):
            print(f"Cross-validation fold {fold + 1}/{k_folds}")
            
            # Create fold data loaders
            train_data = all_data[train_idx]
            train_targets = all_targets[train_idx]
            val_data = all_data[val_idx]
            val_targets = all_targets[val_idx]
            
            # Create temporary data loaders
            from torch.utils.data import TensorDataset, DataLoader
            
            train_dataset = TensorDataset(train_data, train_targets)
            val_dataset = TensorDataset(val_data, val_targets)
            
            fold_train_loader = DataLoader(
                train_dataset, 
                batch_size=train_loader.batch_size,
                shuffle=True
            )
            fold_val_loader = DataLoader(
                val_dataset,
                batch_size=train_loader.batch_size,
                shuffle=False
            )
            
            # Train model on fold
            from .trainer import Trainer
            
            # Clone model for this fold
            fold_model = type(self.model)(self.model.config)
            fold_model.to(self.device)
            
            trainer = Trainer(
                model=fold_model,
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                config={**self.config, 'epochs': self.config.get('cv_epochs', 20)}
            )
            
            trainer.train()
            
            # Evaluate fold
            fold_evaluator = Evaluator(fold_model, self.config)
            fold_results = fold_evaluator.evaluate(fold_val_loader, save_results=False)
            
            cv_results.append(fold_results['metrics'])
        
        # Aggregate results
        cv_metrics = self._aggregate_cv_results(cv_results)
        
        return {
            'cv_metrics': cv_metrics,
            'fold_results': cv_results
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate cross-validation results across folds.
        
        Args:
            fold_results: Results from each fold
            
        Returns:
            Aggregated metrics with mean and std
        """
        # Extract numeric metrics
        numeric_metrics = {}
        for fold_result in fold_results:
            for key, value in fold_result.items():
                if isinstance(value, (int, float)) and key not in ['confusion_matrix', 'classification_report']:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Compute mean and std
        aggregated = {}
        for metric, values in numeric_metrics.items():
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated
    
    def benchmark_inference(self, test_loader: DataLoader, 
                          num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            test_loader: Test data loader
            num_runs: Number of benchmark runs
            
        Returns:
            Inference benchmark results
        """
        import time
        
        self.model.eval()
        
        # Warm up
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                _ = self.model(data)
                break
        
        # Benchmark
        times = []
        
        with torch.no_grad():
            for run in range(num_runs):
                start_time = time.time()
                
                for data, _ in test_loader:
                    data = data.to(self.device)
                    _ = self.model(data)
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times))
        }
    
    def _save_results(self) -> None:
        """Save evaluation results to disk."""
        output_dir = Path(self.config.get('output_dir', 'experiments'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = {}
            for key, value in self.results['metrics'].items():
                if isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                else:
                    metrics_json[key] = value
            json.dump(metrics_json, f, indent=2)
        
        # Save predictions and targets
        results_df = pd.DataFrame({
            'predictions': self.results['predictions'],
            'targets': self.results['targets']
        })
        results_df.to_csv(output_dir / 'predictions.csv', index=False)
        
        print(f"Evaluation results saved to {output_dir}")
    
    def get_feature_importance(self, test_loader: DataLoader, 
                             method: str = 'gradient') -> Dict[str, Any]:
        """
        Compute feature importance using various methods.
        
        Args:
            test_loader: Test data loader
            method: Importance computation method ('gradient')
            
        Returns:
            Feature importance scores and metadata
        """
        try:
            if method == 'gradient':
                return self._gradient_based_importance(test_loader)
            else:
                # For now, only gradient method is implemented
                print(f"Warning: Method '{method}' not implemented, using gradient method")
                return self._gradient_based_importance(test_loader)
        except Exception as e:
            print(f"Warning: Feature importance computation failed: {e}")
            return {
                'feature_importance': np.array([]),
                'method': method,
                'error': str(e)
            }
    
    def _gradient_based_importance(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Compute gradient-based feature importance."""
        self.model.eval()
        
        importances = []
        
        for data, target in test_loader:
            data = data.to(self.device).requires_grad_(True)
            target = target.to(self.device)
            
            output = self.model(data)
            
            if self.task_type == 'classification':
                # Use the predicted class probability
                loss = output.max(1)[0].sum()
            else:
                loss = output.sum()
            
            loss.backward()
            
            # Get gradients
            gradients = data.grad.abs().mean(0)
            importances.append(gradients.cpu().numpy())
        
        importance_scores = np.mean(importances, axis=0)
        
        return {
            'feature_importance': importance_scores,
            'method': 'gradient'
        }

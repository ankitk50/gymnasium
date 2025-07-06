"""
Evaluation visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from sklearn.metrics import confusion_matrix
import itertools


class EvaluationVisualizer:
    """
    Visualizer for model evaluation results.
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize the evaluation visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, cm: Union[np.ndarray, List[List]], 
                            class_names: Optional[List[str]] = None,
                            title: str = 'Confusion Matrix',
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Class names for labels
            title: Plot title
            save_path: Path to save the plot
        """
        cm = np.array(cm)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot raw confusion matrix
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.figure.colorbar(im1, ax=ax1)
        ax1.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names or range(cm.shape[1]),
                yticklabels=class_names or range(cm.shape[0]),
                title=f'{title} (Raw Counts)',
                ylabel='True label',
                xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        # Plot normalized confusion matrix
        im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.figure.colorbar(im2, ax=ax2)
        ax2.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names or range(cm.shape[1]),
                yticklabels=class_names or range(cm.shape[0]),
                title=f'{title} (Normalized)',
                ylabel='True label',
                xlabel='Predicted label')
        
        # Add text annotations for normalized matrix
        thresh_norm = cm_normalized.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax2.text(j, i, format(cm_normalized[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm_normalized[i, j] > thresh_norm else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_classification_report(self, report: Dict[str, Any],
                                 save_path: Optional[str] = None) -> None:
        """
        Plot classification report as heatmap.
        
        Args:
            report: Classification report dictionary
            save_path: Path to save the plot
        """
        # Extract metrics for each class
        classes = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']
        
        # Create matrix
        data = []
        for cls in classes:
            if isinstance(report[cls], dict):
                row = [report[cls].get(metric, 0) for metric in metrics]
                data.append(row)
        
        if not data:
            return
        
        df = pd.DataFrame(data, index=classes, columns=metrics)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, cmap='Blues', fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Classification Report Heatmap')
        plt.ylabel('Classes')
        plt.xlabel('Metrics')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'classification_report.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                              metrics: Dict[str, float],
                              save_path: Optional[str] = None) -> None:
        """
        Plot regression results including scatter plot and residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Regression metrics
            save_path: Path to save the plot
        """
        # Ensure arrays are properly shaped and aligned
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        if len(y_true) == 0 or len(y_pred) == 0:
            print("Warning: Empty arrays provided to plot_regression_results")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Predicted vs True
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Predicted vs True (R² = {metrics.get("r2_score", 0):.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for residuals normality
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')
            axes[1, 1].grid(True, alpha=0.3)
        except ImportError:
            # If scipy is not available, create a simple scatter plot
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
            axes[1, 1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
            axes[1, 1].set_xlabel('Theoretical Quantiles')
            axes[1, 1].set_ylabel('Sample Quantiles')
            axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f"""Metrics:
MAE: {metrics.get('mae', 0):.4f}
MSE: {metrics.get('mse', 0):.4f}
RMSE: {metrics.get('rmse', 0):.4f}
R²: {metrics.get('r2_score', 0):.4f}
MAPE: {metrics.get('mape', 0):.2f}%"""
        
        fig.text(0.02, 0.98, metrics_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'regression_results.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Plot detailed residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        # Ensure arrays are flattened and same shape
        y_true_flat = np.asarray(y_true).flatten()
        y_pred_flat = np.asarray(y_pred).flatten()
        
        # Ensure same length
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        residuals = y_true_flat - y_pred_flat
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_pred_flat, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Absolute residuals vs Fitted
        abs_residuals = np.abs(residuals)
        axes[0, 1].scatter(y_pred_flat, abs_residuals, alpha=0.6)
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('|Residuals|')
        axes[0, 1].set_title('Absolute Residuals vs Fitted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 0].axvline(x=residuals.mean(), color='r', linestyle='--', 
                          label=f'Mean: {residuals.mean():.4f}')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Order (to check for patterns)
        order = np.arange(len(residuals))
        axes[1, 1].scatter(order, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Observation Order')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Order')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_feature_importance(self, importance_scores: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              title: str = 'Feature Importance',
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance scores.
        
        Args:
            importance_scores: Feature importance scores
            feature_names: Names of features
            title: Plot title
            save_path: Path to save the plot
        """
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance_scores))]
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)
        sorted_scores = importance_scores[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(sorted_scores)), sorted_scores)
        plt.yticks(range(len(sorted_scores)), sorted_names)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Color bars by importance
        try:
            cmap = plt.cm.get_cmap('viridis')
            colors = cmap(sorted_scores / sorted_scores.max())
        except (AttributeError, ValueError):
            # Fallback to basic colors
            colors = ['skyblue'] * len(sorted_scores)
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot distribution comparison of true vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_true, bins=30, alpha=0.7, label='True', density=True)
        plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        box_plot = plt.boxplot([y_true, y_pred])
        plt.xticks([1, 2], ['True', 'Predicted'])
        plt.ylabel('Value')
        plt.title('Box Plot Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_interactive_evaluation_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray,
                                              metrics: Dict[str, float]) -> None:
        """
        Create interactive evaluation dashboard using Plotly.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Evaluation metrics
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predicted vs True', 'Residuals', 'Distribution Comparison', 'Metrics'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "table"}]]
        )
        
        # Scatter plot: Predicted vs True
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                      marker=dict(size=5, opacity=0.6)),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residuals plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                      marker=dict(size=5, opacity=0.6)),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_trace(
            go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0],
                      mode='lines', name='Zero Line',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        # Distribution comparison
        fig.add_trace(
            go.Histogram(x=y_true, name='True Values', opacity=0.7, nbinsx=30),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=y_pred, name='Predicted Values', opacity=0.7, nbinsx=30),
            row=2, col=1
        )
        
        # Metrics table
        metrics_data = [[k, f"{v:.4f}"] for k, v in metrics.items()]
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*metrics_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive Evaluation Dashboard")
        fig.write_html(self.output_dir / 'evaluation_dashboard.html')
    
    def plot_learning_curves(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                           val_scores: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot learning curves showing performance vs training set size.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            save_path: Path to save the plot
        """
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.xlabel("Training Set Size")
        plt.ylabel("Score")
        plt.title("Learning Curves")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()

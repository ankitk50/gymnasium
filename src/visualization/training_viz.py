"""
Training visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch


class TrainingVisualizer:
    """
    Visualizer for training metrics and progress.
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize the training visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_curves(self, history: Dict[str, List[float]], 
                           save_path: Optional[str] = None) -> None:
        """
        Plot training and validation curves.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Loss curves
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if history.get('val_loss') and len(history['val_loss']) > 0:
            val_epochs = range(1, len(history['val_loss']) + 1)
            axes[0, 0].plot(val_epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in history:
            axes[0, 1].plot(epochs[:len(history['lr'])], history['lr'], 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Training metrics
        if history.get('train_metrics'):
            self._plot_metrics(axes[1, 0], history['train_metrics'], 'Training Metrics', epochs)
        
        # Validation metrics
        if history.get('val_metrics') and len(history['val_metrics']) > 0:
            val_epochs = range(1, len(history['val_metrics']) + 1)
            self._plot_metrics(axes[1, 1], history['val_metrics'], 'Validation Metrics', val_epochs)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_metrics(self, ax, metrics_list: List[Dict[str, float]], 
                     title: str, epochs: range) -> None:
        """Plot metrics on given axis."""
        if not metrics_list:
            return
        
        # Extract metric names (excluding 'loss' as it's plotted separately)
        metric_names = [key for key in metrics_list[0].keys() if key != 'loss']
        
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in metrics_list]
            ax.plot(epochs[:len(values)], values, label=metric_name, linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_interactive_training_curves(self, history: Dict[str, List[float]]) -> None:
        """
        Create interactive training curves using Plotly.
        
        Args:
            history: Training history dictionary
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Learning Rate', 'Training Metrics', 'Validation Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        if history.get('val_loss') and len(history['val_loss']) > 0:
            val_epochs = list(range(1, len(history['val_loss']) + 1))
            fig.add_trace(
                go.Scatter(x=val_epochs, y=history['val_loss'], name='Val Loss',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
        
        # Learning rate
        if 'lr' in history:
            fig.add_trace(
                go.Scatter(x=epochs[:len(history['lr'])], y=history['lr'], name='Learning Rate',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive Training Progress",
            showlegend=True
        )
        
        # Update y-axis for learning rate to log scale
        fig.update_yaxes(type="log", row=1, col=2)
        
        # Save interactive plot
        fig.write_html(self.output_dir / 'interactive_training_curves.html')
    
    def plot_gradient_flow(self, model: torch.nn.Module) -> None:
        """
        Plot gradient flow through the model.
        
        Args:
            model: PyTorch model
        """
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                layers.append(name)
                ave_grads.append(parameter.grad.abs().mean().cpu())
                max_grads.append(parameter.grad.abs().max().cpu())
        
        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, label="max-gradient")
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.6, label="mean-gradient")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # Zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'gradient_flow.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weight_distributions(self, model: torch.nn.Module) -> None:
        """
        Plot weight distributions for each layer.
        
        Args:
            model: PyTorch model
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        weights = []
        layer_names = []
        
        for name, parameter in model.named_parameters():
            if 'weight' in name and parameter.dim() > 1:
                weights.append(parameter.detach().cpu().numpy().flatten())
                layer_names.append(name)
        
        # Plot histograms
        for i, (weight, name) in enumerate(zip(weights, layer_names)):
            axes[0].hist(weight, bins=50, alpha=0.7, label=name, density=True)
        
        axes[0].set_title('Weight Distributions')
        axes[0].set_xlabel('Weight Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot weight statistics
        means = [np.mean(w) for w in weights]
        stds = [np.std(w) for w in weights]
        
        x_pos = np.arange(len(layer_names))
        axes[1].bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7)
        axes[1].bar(x_pos + 0.2, stds, 0.4, label='Std', alpha=0.7)
        axes[1].set_title('Weight Statistics by Layer')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Value')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([name.split('.')[-2] for name in layer_names], rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weight_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_loss_landscape(self, loss_values: np.ndarray, 
                          x_range: np.ndarray, y_range: np.ndarray) -> None:
        """
        Plot loss landscape (if available).
        
        Args:
            loss_values: 2D array of loss values
            x_range: X-axis parameter range
            y_range: Y-axis parameter range
        """
        fig = plt.figure(figsize=(12, 5))
        
        # 2D contour plot
        ax1 = fig.add_subplot(121)
        contour = ax1.contour(x_range, y_range, loss_values, levels=20)
        ax1.clabel(contour, inline=True, fontsize=8)
        ax1.set_title('Loss Landscape (Contour)')
        ax1.set_xlabel('Parameter 1')
        ax1.set_ylabel('Parameter 2')
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        X, Y = np.meshgrid(x_range, y_range)
        surface = ax2.plot_surface(X, Y, loss_values, cmap='viridis', alpha=0.8)
        ax2.set_title('Loss Landscape (3D)')
        ax2.set_xlabel('Parameter 1')
        ax2.set_ylabel('Parameter 2')
        ax2.set_zlabel('Loss')
        
        plt.colorbar(surface, ax=ax2, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_landscape.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def update_training_plots(self, history: Dict[str, List[float]]) -> None:
        """
        Update training plots during training.
        
        Args:
            history: Training history dictionary
        """
        # Update standard plots
        self.plot_training_curves(history)
        
        # Update interactive plots
        self.plot_interactive_training_curves(history)
    
    def create_training_summary(self, history: Dict[str, List[float]], 
                              model_info: Dict[str, Any]) -> None:
        """
        Create a comprehensive training summary report.
        
        Args:
            history: Training history
            model_info: Model information
        """
        # Create summary statistics
        final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
        final_val_loss = history['val_loss'][-1] if history.get('val_loss') else 0
        best_val_loss = min(history['val_loss']) if history.get('val_loss') else 0
        
        summary_text = f"""
# Training Summary Report

## Model Information
- **Model Name**: {model_info.get('name', 'Unknown')}
- **Parameters**: {model_info.get('num_parameters', 0):,}
- **Device**: {model_info.get('device', 'Unknown')}

## Training Results
- **Total Epochs**: {len(history['train_loss'])}
- **Final Training Loss**: {final_train_loss:.6f}
- **Final Validation Loss**: {final_val_loss:.6f}
- **Best Validation Loss**: {best_val_loss:.6f}

## Training Configuration
{self._format_config(model_info.get('config', {}))}
        """
        
        # Save summary
        with open(self.output_dir / 'training_summary.md', 'w') as f:
            f.write(summary_text)
    
    def _format_config(self, config: Dict[str, Any], indent: int = 0) -> str:
        """Format configuration dictionary for markdown."""
        result = ""
        for key, value in config.items():
            if isinstance(value, dict):
                result += "  " * indent + f"- **{key}**:\n"
                result += self._format_config(value, indent + 1)
            else:
                result += "  " * indent + f"- **{key}**: {value}\n"
        return result

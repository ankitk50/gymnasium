"""
Inference visualization utilities.
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
import torch


class InferenceVisualizer:
    """
    Visualizer for model inference results and real-time monitoring.
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize the inference visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_cpu_allocation_timeline(self, timestamps: List, 
                                   allocations: List[float],
                                   actual_usage: Optional[List[float]] = None,
                                   task_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot CPU allocation over time.
        
        Args:
            timestamps: Time points
            allocations: Predicted CPU allocations
            actual_usage: Actual CPU usage (optional)
            task_names: Task names for labeling
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 8))
        
        # Convert to numpy arrays for easier handling
        timestamps = np.array(timestamps)
        allocations = np.array(allocations)
        
        # Plot predicted allocations
        plt.plot(timestamps, allocations, 'b-', linewidth=2, label='Predicted Allocation', marker='o')
        
        # Plot actual usage if available
        if actual_usage is not None:
            actual_usage = np.array(actual_usage)
            plt.plot(timestamps, actual_usage, 'r-', linewidth=2, label='Actual Usage', marker='s')
            
            # Fill between for visualization of over/under allocation
            over_allocation = np.maximum(allocations - actual_usage, 0)
            under_allocation = np.maximum(actual_usage - allocations, 0)
            
            plt.fill_between(timestamps, allocations, actual_usage, 
                           where=(allocations >= actual_usage), 
                           alpha=0.3, color='orange', label='Over-allocation')
            plt.fill_between(timestamps, allocations, actual_usage, 
                           where=(allocations < actual_usage), 
                           alpha=0.3, color='red', label='Under-allocation')
        
        # Add task name annotations if provided
        if task_names:
            for i, (t, a, name) in enumerate(zip(timestamps, allocations, task_names)):
                if i % 5 == 0:  # Annotate every 5th point to avoid clutter
                    plt.annotate(name, (t, a), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Time')
        plt.ylabel('CPU Allocation (0-1)')
        plt.title('CPU Allocation Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'cpu_allocation_timeline.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_allocation_heatmap(self, allocation_matrix: np.ndarray,
                              task_names: Optional[List[str]] = None,
                              time_labels: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot CPU allocation as a heatmap across tasks and time.
        
        Args:
            allocation_matrix: Matrix of allocations (tasks x time)
            task_names: Names of tasks
            time_labels: Time point labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Create labels if not provided
        if task_names is None:
            task_names = [f'Task {i}' for i in range(allocation_matrix.shape[0])]
        if time_labels is None:
            time_labels = [f'T{i}' for i in range(allocation_matrix.shape[1])]
        
        # Create heatmap
        sns.heatmap(allocation_matrix, 
                   xticklabels=time_labels,
                   yticklabels=task_names,
                   cmap='YlOrRd',
                   annot=True if allocation_matrix.size < 100 else False,
                   fmt='.2f',
                   cbar_kws={'label': 'CPU Allocation'})
        
        plt.title('CPU Allocation Heatmap')
        plt.xlabel('Time')
        plt.ylabel('Tasks')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'allocation_heatmap.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_resource_utilization(self, timestamps: List,
                                cpu_utilization: List[float],
                                memory_utilization: List[float],
                                io_utilization: Optional[List[float]] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot system resource utilization over time.
        
        Args:
            timestamps: Time points
            cpu_utilization: CPU utilization values
            memory_utilization: Memory utilization values
            io_utilization: I/O utilization values (optional)
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(3 if io_utilization else 2, 1, figsize=(15, 10))
        
        # CPU utilization
        axes[0].plot(timestamps, cpu_utilization, 'b-', linewidth=2, marker='o')
        axes[0].set_ylabel('CPU Utilization (%)')
        axes[0].set_title('System Resource Utilization')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='High Usage Threshold')
        axes[0].legend()
        
        # Memory utilization
        axes[1].plot(timestamps, memory_utilization, 'g-', linewidth=2, marker='s')
        axes[1].set_ylabel('Memory Utilization (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=85, color='r', linestyle='--', alpha=0.7, label='High Usage Threshold')
        axes[1].legend()
        
        # I/O utilization (if provided)
        if io_utilization:
            axes[2].plot(timestamps, io_utilization, 'm-', linewidth=2, marker='^')
            axes[2].set_ylabel('I/O Utilization (%)')
            axes[2].set_xlabel('Time')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='High Usage Threshold')
            axes[2].legend()
        else:
            axes[1].set_xlabel('Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'resource_utilization.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_task_performance_metrics(self, task_data: Dict[str, Dict[str, float]],
                                    save_path: Optional[str] = None) -> None:
        """
        Plot task performance metrics comparison.
        
        Args:
            task_data: Dictionary of task names to metrics
            save_path: Path to save the plot
        """
        # Extract data for plotting
        tasks = list(task_data.keys())
        metrics = list(next(iter(task_data.values())).keys())
        
        # Create subplot for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            values = [task_data[task][metric] for task in tasks]
            
            bars = axes[i].bar(tasks, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Color bars by value
            max_val = max(values)
            min_val = min(values)
            if max_val != min_val:
                colors = plt.cm.viridis([(v - min_val) / (max_val - min_val) for v in values])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        # Hide unused subplots
        for j in range(n_metrics, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'task_performance_metrics.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_real_time_dashboard(self, data: Dict[str, Any]) -> str:
        """
        Create a real-time monitoring dashboard using Plotly.
        
        Args:
            data: Real-time data dictionary
            
        Returns:
            Path to saved HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('CPU Allocation', 'System CPU Usage', 'Memory Usage', 
                          'Task Queue', 'Throughput', 'Latency'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        timestamps = data.get('timestamps', [])
        
        # CPU Allocation timeline
        if 'cpu_allocations' in data:
            fig.add_trace(
                go.Scatter(x=timestamps, y=data['cpu_allocations'], 
                          mode='lines+markers', name='CPU Allocation',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # System CPU Usage
        if 'system_cpu' in data:
            fig.add_trace(
                go.Scatter(x=timestamps, y=data['system_cpu'], 
                          mode='lines+markers', name='System CPU',
                          line=dict(color='red', width=2)),
                row=1, col=2
            )
        
        # Memory Usage
        if 'memory_usage' in data:
            fig.add_trace(
                go.Scatter(x=timestamps, y=data['memory_usage'], 
                          mode='lines+markers', name='Memory Usage',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # Task Queue Length
        if 'queue_length' in data:
            fig.add_trace(
                go.Bar(x=timestamps[-10:], y=data['queue_length'][-10:], 
                      name='Queue Length'),
                row=2, col=2
            )
        
        # Throughput
        if 'throughput' in data:
            fig.add_trace(
                go.Scatter(x=timestamps, y=data['throughput'], 
                          mode='lines+markers', name='Throughput',
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
        
        # Latency
        if 'latency' in data:
            fig.add_trace(
                go.Scatter(x=timestamps, y=data['latency'], 
                          mode='lines+markers', name='Latency',
                          line=dict(color='orange', width=2)),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Real-time CPU Allocation Dashboard",
            showlegend=False
        )
        
        # Add threshold lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", opacity=0.7, row=1, col=2)
        fig.add_hline(y=0.85, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        
        # Save dashboard
        dashboard_path = self.output_dir / 'real_time_dashboard.html'
        fig.write_html(dashboard_path)
        
        return str(dashboard_path)
    
    def plot_prediction_confidence(self, predictions: np.ndarray,
                                 confidence_intervals: np.ndarray,
                                 timestamps: Optional[List] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot predictions with confidence intervals.
        
        Args:
            predictions: Model predictions
            confidence_intervals: Confidence intervals (lower, upper)
            timestamps: Time points (optional)
            save_path: Path to save the plot
        """
        if timestamps is None:
            timestamps = range(len(predictions))
        
        plt.figure(figsize=(15, 8))
        
        # Plot predictions
        plt.plot(timestamps, predictions, 'b-', linewidth=2, label='Predictions', marker='o')
        
        # Plot confidence intervals
        lower_bound = confidence_intervals[:, 0]
        upper_bound = confidence_intervals[:, 1]
        
        plt.fill_between(timestamps, lower_bound, upper_bound, 
                        alpha=0.3, color='blue', label='Confidence Interval')
        
        plt.xlabel('Time')
        plt.ylabel('CPU Allocation')
        plt.title('Predictions with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'prediction_confidence.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_allocation_efficiency(self, efficiency_data: Dict[str, List[float]],
                                 save_path: Optional[str] = None) -> None:
        """
        Plot allocation efficiency metrics over time.
        
        Args:
            efficiency_data: Dictionary of efficiency metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['allocation_accuracy', 'resource_utilization', 'waste_ratio', 'under_allocation_rate']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in efficiency_data:
                values = efficiency_data[metric]
                timestamps = range(len(values))
                
                axes[i].plot(timestamps, values, color=color, linewidth=2, marker='o')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(timestamps, values, 1)
                p = np.poly1d(z)
                axes[i].plot(timestamps, p(timestamps), "--", color='gray', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'allocation_efficiency.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def export_inference_report(self, inference_data: Dict[str, Any],
                              model_info: Dict[str, Any]) -> str:
        """
        Export comprehensive inference report.
        
        Args:
            inference_data: Inference results and metrics
            model_info: Model information
            
        Returns:
            Path to saved report
        """
        report_content = f"""
# CPU Allocation Inference Report

## Model Information
- **Model Type**: {model_info.get('name', 'Unknown')}
- **Parameters**: {model_info.get('num_parameters', 0):,}
- **Device**: {model_info.get('device', 'Unknown')}

## Inference Summary
- **Total Predictions**: {len(inference_data.get('predictions', []))}
- **Average Allocation**: {np.mean(inference_data.get('predictions', [0])):.4f}
- **Allocation Range**: {np.min(inference_data.get('predictions', [0])):.4f} - {np.max(inference_data.get('predictions', [0])):.4f}

## Performance Metrics
"""
        
        if 'efficiency_metrics' in inference_data:
            metrics = inference_data['efficiency_metrics']
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report_content += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
        
        report_content += f"""

## Visualizations Generated
- CPU Allocation Timeline
- Resource Utilization Charts
- Allocation Heatmap
- Performance Metrics Dashboard

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        report_path = self.output_dir / 'inference_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return str(report_path)

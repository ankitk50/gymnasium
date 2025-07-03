"""
Logging utilities for the training pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import time


def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[Union[str, Path]] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file to log to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    
    # Configure root logger
    logger = logging.getLogger('ml_pipeline')
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """
    Specialized logger for training pipeline with additional functionality.
    """
    
    def __init__(self, name: str = 'training', log_file: Optional[Path] = None):
        """
        Initialize training logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.start_time = time.time()
        self.epoch_start_time = None
        
        # Setup file logging if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_training_start(self, model_info: dict, data_info: dict) -> None:
        """
        Log training start information.
        
        Args:
            model_info: Model information dictionary
            data_info: Data information dictionary
        """
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 50)
        
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("Data Information:")
        for key, value in data_info.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 50)
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """
        Log epoch start.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
        """
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch + 1}/{total_epochs} started")
    
    def log_epoch_end(self, epoch: int, train_metrics: dict, val_metrics: dict = None) -> None:
        """
        Log epoch end with metrics.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            
            log_msg = f"Epoch {epoch + 1} completed in {epoch_time:.2f}s"
            
            # Add training metrics
            train_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            log_msg += f" | Train: {train_str}"
            
            # Add validation metrics if available
            if val_metrics:
                val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                log_msg += f" | Val: {val_str}"
            
            self.logger.info(log_msg)
    
    def log_training_end(self, total_epochs: int, best_metrics: dict = None) -> None:
        """
        Log training completion.
        
        Args:
            total_epochs: Total epochs completed
            best_metrics: Best validation metrics (optional)
        """
        total_time = time.time() - self.start_time
        
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Average time per epoch: {total_time/total_epochs:.2f}s")
        
        if best_metrics:
            self.logger.info("Best validation metrics:")
            for key, value in best_metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
        
        self.logger.info("=" * 50)
    
    def log_checkpoint_saved(self, epoch: int, checkpoint_path: Path) -> None:
        """
        Log checkpoint save.
        
        Args:
            epoch: Current epoch
            checkpoint_path: Path to saved checkpoint
        """
        self.logger.info(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")
    
    def log_early_stopping(self, epoch: int, patience: int) -> None:
        """
        Log early stopping.
        
        Args:
            epoch: Current epoch
            patience: Patience value
        """
        self.logger.warning(f"Early stopping triggered at epoch {epoch + 1} "
                          f"(patience: {patience})")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log error with context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        if context:
            self.logger.error(f"Error in {context}: {str(error)}")
        else:
            self.logger.error(f"Error: {str(error)}")
        
        # Log traceback for debugging
        import traceback
        self.logger.debug(traceback.format_exc())


class MetricsLogger:
    """
    Logger specifically for metrics tracking and reporting.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Optional log file for metrics
        """
        self.logger = logging.getLogger('metrics')
        self.metrics_history = []
        
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create CSV-style handler for metrics
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')  # Plain format for CSV
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Log header
            self.logger.info("timestamp,epoch,phase,metric,value")
    
    def log_metric(self, epoch: int, phase: str, metric_name: str, value: float) -> None:
        """
        Log a single metric value.
        
        Args:
            epoch: Current epoch
            phase: Training phase ('train', 'val', 'test')
            metric_name: Name of the metric
            value: Metric value
        """
        timestamp = time.time()
        self.metrics_history.append({
            'timestamp': timestamp,
            'epoch': epoch,
            'phase': phase,
            'metric': metric_name,
            'value': value
        })
        
        # Log to file if handler exists
        if self.logger.handlers:
            self.logger.info(f"{timestamp},{epoch},{phase},{metric_name},{value}")
    
    def log_metrics_dict(self, epoch: int, phase: str, metrics: dict) -> None:
        """
        Log multiple metrics from a dictionary.
        
        Args:
            epoch: Current epoch
            phase: Training phase
            metrics: Dictionary of metrics
        """
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(epoch, phase, metric_name, value)
    
    def get_metric_history(self, metric_name: str, phase: str = None) -> list:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            phase: Optional phase filter
            
        Returns:
            List of metric values
        """
        filtered_history = [
            entry for entry in self.metrics_history
            if entry['metric'] == metric_name and (phase is None or entry['phase'] == phase)
        ]
        return [entry['value'] for entry in filtered_history]


class ExperimentLogger:
    """
    High-level experiment logger that combines training and metrics logging.
    """
    
    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Output directory for logs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self.training_logger = TrainingLogger(
            name=f'{experiment_name}_training',
            log_file=self.output_dir / 'training.log'
        )
        
        self.metrics_logger = MetricsLogger(
            log_file=self.output_dir / 'metrics.csv'
        )
        
        # Main logger
        self.logger = logging.getLogger(f'{experiment_name}_experiment')
    
    def log_experiment_start(self, config: dict) -> None:
        """
        Log experiment start with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Log configuration
        self.logger.info("Experiment configuration:")
        self._log_dict(config, indent=1)
    
    def log_experiment_end(self, results: dict) -> None:
        """
        Log experiment completion with results.
        
        Args:
            results: Final experiment results
        """
        self.logger.info(f"Experiment {self.experiment_name} completed")
        self.logger.info("Final results:")
        self._log_dict(results, indent=1)
    
    def _log_dict(self, d: dict, indent: int = 0) -> None:
        """
        Recursively log dictionary contents.
        
        Args:
            d: Dictionary to log
            indent: Indentation level
        """
        for key, value in d.items():
            if isinstance(value, dict):
                self.logger.info("  " * indent + f"{key}:")
                self._log_dict(value, indent + 1)
            else:
                self.logger.info("  " * indent + f"{key}: {value}")


# Global logger instance
_global_logger = None


def get_logger(name: str = 'ml_pipeline') -> logging.Logger:
    """
    Get or create a global logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return logging.getLogger(name)

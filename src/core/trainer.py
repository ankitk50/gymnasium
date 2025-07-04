"""
Core trainer module for the training pipeline framework.
Handles the training loop, validation, and logging.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Wandb integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .base_model import BaseModel
from ..utils.metrics import MetricsTracker
from ..visualization.training_viz import TrainingVisualizer


class Trainer:
    """
    Generic trainer class for training models with the pipeline framework.
    """
    
    def __init__(self, 
                 model: BaseModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None,
                 use_wandb: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            logger: Logger instance
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Wandb integration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            # Watch model with wandb
            wandb.watch(self.model, log_freq=100)
            self.logger.info("Wandb integration enabled")
        
        # Training parameters
        self.epochs = self.config.get('epochs', 100)
        self.save_every = self.config.get('save_every', 10)
        self.validate_every = self.config.get('validate_every', 1)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.gradient_clip_val = self.config.get('gradient_clip_val', None)
        
        # Setup training components
        self.device = self.model.device
        self.model.to(self.device)
        
        self.criterion = model.get_loss_function()
        self.optimizer = model.get_optimizer(model.parameters())
        self.scheduler = model.get_scheduler(self.optimizer)
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.visualizer = TrainingVisualizer(self.config.get('output_dir', 'experiments'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        epoch_metrics = self.metrics_tracker.reset()
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                
                # Update metrics
                batch_metrics = self._compute_batch_metrics(output, target, loss)
                epoch_metrics = self.metrics_tracker.update(batch_metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Validation metrics for the epoch
        """
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        epoch_metrics = self.metrics_tracker.reset()
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Update metrics
                batch_metrics = self._compute_batch_metrics(output, target, loss)
                epoch_metrics = self.metrics_tracker.update(batch_metrics)
        
        return epoch_metrics
    
    def _compute_batch_metrics(self, output: torch.Tensor, target: torch.Tensor, 
                              loss: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for a batch.
        
        Args:
            output: Model output
            target: Ground truth
            loss: Loss value
            
        Returns:
            Batch metrics dictionary
        """
        metrics = {'loss': loss.item()}
        
        # Add task-specific metrics based on problem type
        if self.config.get('task_type') == 'classification':
            pred = output.argmax(dim=1)
            accuracy = (pred == target).float().mean().item()
            metrics['accuracy'] = accuracy
            
        elif self.config.get('task_type') == 'regression':
            mse = nn.MSELoss()(output, target).item()
            mae = nn.L1Loss()(output, target).item()
            metrics.update({'mse': mse, 'mae': mae})
            
        return metrics
    
    def train(self) -> BaseModel:
        """
        Main training loop.
        
        Returns:
            Trained model
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Model has {self.model.count_parameters()} trainable parameters")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validate epoch
            val_metrics = {}
            if self.val_loader and epoch % self.validate_every == 0:
                val_metrics = self.validate_epoch()
                self.training_history['val_loss'].append(val_metrics.get('loss', 0.0))
                self.training_history['val_metrics'].append(val_metrics)
                
                # Early stopping check
                if self._should_early_stop(val_metrics.get('loss', float('inf'))):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Logging
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Update visualizations
            self.visualizer.update_training_plots(self.training_history)
        
        # Final save
        self._save_checkpoint(self.current_epoch, train_metrics, val_metrics, is_final=True)
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Log final metrics to wandb
        if self.use_wandb:
            final_metrics = {
                "training_time_seconds": training_time,
                "total_epochs": self.current_epoch + 1,
                "best_val_loss": self.best_val_loss
            }
            wandb.log(final_metrics)
        
        return self.model
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """
        Check if training should be stopped early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.early_stopping_patience
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float]) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        log_str = f"Epoch {epoch:3d} | "
        log_str += f"Train Loss: {train_metrics['loss']:.4f} | "
        
        if val_metrics:
            log_str += f"Val Loss: {val_metrics['loss']:.4f} | "
        
        log_str += f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        
        self.logger.info(log_str)
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb_log = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            wandb.log(wandb_log)
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict[str, float], 
                        val_metrics: Dict[str, float], is_final: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            is_final: Whether this is the final checkpoint
        """
        output_dir = Path(self.config.get('output_dir', 'experiments'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = 'final_model.pt' if is_final else f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path = output_dir / checkpoint_name
        
        metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
        
        self.model.save_checkpoint(
            checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=metrics
        )
        
        # Also save best model
        if val_metrics and val_metrics.get('loss', float('inf')) <= self.best_val_loss:
            best_model_path = output_dir / 'best_model.pt'
            self.model.save_checkpoint(
                best_model_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics
            )
    
    def get_training_history(self) -> Dict[str, List]:
        """
        Get the training history.
        
        Returns:
            Training history dictionary
        """
        return self.training_history

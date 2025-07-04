"""
Train a model with Weights & Biases (wandb) integration for visualization.
"""

import sys
from pathlib import Path
import torch
import wandb
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.core.pipeline import TrainingPipeline
from src.utils.config import create_default_config
from src.data.data_loader import create_synthetic_cpu_allocation_data


def train_with_wandb():
    """Train a model with wandb logging and visualization."""
    
    # Initialize wandb
    wandb.init(
        project="gymnasium-ml-pipeline",
        name="cpu-allocation-experiment",
        config={
            "architecture": "MLP",
            "dataset": "synthetic_cpu_allocation",
            "epochs": 50,
        }
    )
    
    print("🚀 Starting training with Weights & Biases integration")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config()
    config.set('experiment_name', 'wandb_experiment')
    config.set('training.epochs', 50)
    config.set('model.name', 'cpu_allocation_mlp')
    config.set('model.hidden_dims', [128, 64, 32])
    config.set('model.dropout_rate', 0.2)
    config.set('training.learning_rate', 0.001)
    config.set('training.batch_size', 32)
    
    # Log configuration to wandb
    wandb.config.update(config.to_dict())
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        config_dict=config.to_dict(),
        experiment_name='wandb_experiment'
    )
    
    # Setup data and model
    pipeline.setup_data()
    pipeline.setup_model()
    
    # Create enhanced trainer with wandb logging
    trainer = WandbTrainer(
        model=pipeline.model,
        train_loader=pipeline.train_loader,
        val_loader=pipeline.val_loader,
        config=config.get('training', {}),
        logger=pipeline.logger
    )
    
    # Train model
    print("Training model with wandb logging...")
    trained_model = trainer.train()
    
    # Evaluate model
    print("Evaluating model...")
    results = pipeline.evaluate(trained_model)
    
    # Log final results to wandb
    wandb.log({
        "final_train_loss": results['metrics']['loss'],
        "final_mse": results['metrics']['mse'],
        "final_r2_score": results['metrics']['r2_score']
    })
    
    # Log model as wandb artifact
    model_artifact = wandb.Artifact(
        name="cpu_allocation_model",
        type="model",
        description="Trained CPU allocation prediction model"
    )
    
    # Save model to artifact
    if trained_model is not None:
        model_path = pipeline.output_dir / "wandb_model.pt"
        torch.save(trained_model.state_dict(), model_path)
        model_artifact.add_file(str(model_path))
        wandb.log_artifact(model_artifact)
    
    print(f"✅ Training completed! Results saved to: {pipeline.output_dir}")
    print(f"📊 Final MSE: {results['metrics']['mse']:.4f}")
    print(f"📊 Final R²: {results['metrics']['r2_score']:.4f}")
    if wandb.run is not None:
        print(f"🌐 View results in wandb dashboard: {wandb.run.url}")
    else:
        print("🌐 Wandb run completed (offline mode)")
    
    # Finish wandb run
    wandb.finish()


class WandbTrainer:
    """Enhanced trainer with wandb integration."""
    
    def __init__(self, model, train_loader, val_loader, config, logger):
        """Initialize the wandb trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        
        # Training parameters
        self.epochs = config.get('epochs', 50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup training components
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        # Watch model with wandb
        wandb.watch(self.model, log_freq=100)
    
    def train_epoch(self, epoch):
        """Train for one epoch with wandb logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics to wandb
            if batch_idx % 50 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch with wandb logging."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate additional metrics
                mse = torch.nn.functional.mse_loss(output, target)
                mae = torch.nn.functional.l1_loss(output, target)
                
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1
        
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_mse': total_mse / num_batches,
            'val_mae': total_mae / num_batches
        }
        
        return val_metrics
    
    def train(self):
        """Main training loop with wandb logging."""
        self.logger.info(f"Starting training for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Log epoch metrics to wandb
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            log_dict.update(val_metrics)
            
            wandb.log(log_dict)
            
            # Print progress
            log_str = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
            if val_metrics:
                log_str += f" | Val Loss: {val_metrics['val_loss']:.4f}"
                log_str += f" | Val MSE: {val_metrics['val_mse']:.4f}"
            
            self.logger.info(log_str)
            print(log_str)
        
        return self.model


if __name__ == '__main__':
    # Set wandb mode (can be 'online', 'offline', or 'disabled')
    os.environ["WANDB_MODE"] = "offline"  # Change to "online" for cloud sync
    
    try:
        train_with_wandb()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        wandb.finish()

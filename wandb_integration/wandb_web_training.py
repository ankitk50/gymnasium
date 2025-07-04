"""
Simple Wandb Local Dashboard Training with Real-time Web Logging
"""

import os
import subprocess
import time
import webbrowser
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleCPUModel(nn.Module):
    """Simple neural network for CPU allocation prediction."""
    
    def __init__(self, input_dim=10, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def setup_wandb_for_local_web():
    """Setup wandb for local web dashboard viewing."""
    
    print("🌐 SETTING UP WANDB LOCAL WEB DASHBOARD")
    print("=" * 50)
    
    # Use wandb in online mode but with local project
    os.environ["WANDB_MODE"] = "online"
    
    # Try anonymous login for local dashboard
    try:
        print("🔐 Setting up wandb for local web access...")
        
        # Check if already logged in
        result = subprocess.run(['wandb', 'whoami'], capture_output=True, text=True)
        if result.returncode != 0:
            # Try anonymous login
            subprocess.run(['wandb', 'login', '--anonymously'], check=True, timeout=30)
            print("✅ Anonymous login successful")
        else:
            print(f"✅ Already logged in: {result.stdout.strip()}")
            
    except Exception as e:
        print(f"⚠️  Login issue: {e}")
        print("🔄 Falling back to offline mode with sync...")
        os.environ["WANDB_MODE"] = "offline"
    
    return True


def create_synthetic_data(n_samples=1000, n_features=10):
    """Create synthetic CPU allocation data."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    # Normalize the target to simulate CPU allocation percentages
    y = (y - y.min()) / (y.max() - y.min()) * 100
    
    return X, y


def create_data_loaders(X, y, batch_size=32, test_size=0.2):
    """Create PyTorch data loaders."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_with_web_logging():
    """Train model with wandb web dashboard logging."""
    
    # Configuration for training
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,
        "model_name": "CPU_Allocation_WebDashboard",
        "hidden_dims": [128, 64, 32],
        "n_features": 10,
        "n_samples": 1000,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "experiment_type": "web_dashboard_demo"
    }
    
    # Initialize wandb run
    run = wandb.init(
        project="cpu-allocation-web-dashboard",
        name=f"web-experiment-{int(time.time())}",
        config=config,
        tags=["web-dashboard", "cpu-allocation", "real-time"],
        notes="Real-time training with web dashboard visualization"
    )
    
    print("🚀 STARTING CPU ALLOCATION TRAINING")
    print(f"📊 Wandb Run: {run.name}")
    print(f"🆔 Run ID: {run.id}")
    if hasattr(run, 'url') and run.url:
        print(f"🔗 Dashboard URL: {run.url}")
    print("=" * 60)
    
    # Create synthetic data
    print("📊 Generating synthetic CPU allocation data...")
    X, y = create_synthetic_data(
        n_samples=config["n_samples"], 
        n_features=config["n_features"]
    )
    
    # Log dataset info
    wandb.log({
        "dataset/samples": len(X),
        "dataset/features": X.shape[1],
        "dataset/target_mean": float(np.mean(y)),
        "dataset/target_std": float(np.std(y)),
        "dataset/target_range": float(np.max(y) - np.min(y))
    })
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X, y, batch_size=config["batch_size"]
    )
    
    print(f"📦 Training batches: {len(train_loader)}")
    print(f"📦 Test batches: {len(test_loader)}")
    
    # Initialize model
    print("🧠 Creating model...")
    model = SimpleCPUModel(
        input_dim=config["n_features"],
        hidden_dims=config["hidden_dims"]
    )
    
    # Watch model with wandb (logs gradients and weights)
    wandb.watch(model, log_freq=25, log_graph=True)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    
    print(f"🔧 Device: {device}")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Start training with comprehensive logging
    print("\\n🎯 STARTING TRAINING WITH WEB DASHBOARD LOGGING")
    print("📊 Open wandb dashboard to see real-time updates!")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_batches = 0
        
        print(f"\\n📈 Epoch {epoch + 1}/{config['epochs']}")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            mae = torch.mean(torch.abs(output - target))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mae.item()
            train_batches += 1
            
            # Log batch metrics every 15 batches
            if batch_idx % 15 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_mae": mae.item(),
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                    "optimization/learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                loss = criterion(output, target)
                mae = torch.mean(torch.abs(output - target))
                
                val_loss += loss.item()
                val_mae += mae.item()
                val_batches += 1
                
                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_train_loss = train_loss / train_batches
        avg_train_mae = train_mae / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_mae = val_mae / val_batches
        
        # Calculate R²
        pred_array = np.array(predictions)
        target_array = np.array(targets)
        r2 = 1 - (np.sum((target_array - pred_array) ** 2) / 
                 np.sum((target_array - np.mean(target_array)) ** 2))
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Comprehensive epoch logging
        epoch_metrics = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/mae": avg_train_mae,
            "validation/loss": avg_val_loss,
            "validation/mae": avg_val_mae,
            "validation/r2_score": r2,
            "optimization/learning_rate": optimizer.param_groups[0]['lr'],
            "tracking/best_val_loss": best_val_loss,
            "tracking/patience_counter": patience_counter
        }
        
        wandb.log(epoch_metrics)
        
        # Print progress
        print(f"   📊 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   📊 Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f}")
        print(f"   📊 R² Score: {r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Create visualization every 10 epochs
        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            plt.figure(figsize=(10, 4))
            
            # Prediction scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(target_array, pred_array, alpha=0.6, s=20)
            plt.plot([target_array.min(), target_array.max()], 
                    [target_array.min(), target_array.max()], 'r--', lw=2)
            plt.xlabel('Actual CPU Allocation (%)')
            plt.ylabel('Predicted CPU Allocation (%)')
            plt.title(f'Predictions vs Actual (Epoch {epoch})')
            plt.grid(True, alpha=0.3)
            
            # Residuals plot
            plt.subplot(1, 2, 2)
            residuals = target_array - pred_array
            plt.scatter(pred_array, residuals, alpha=0.6, s=20)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot (Epoch {epoch})')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({f"visualizations/epoch_{epoch}_analysis": wandb.Image(plt)})
            plt.close()
    
    # Final summary
    print("\\n✅ TRAINING COMPLETED!")
    print(f"📊 Final Validation Loss: {avg_val_loss:.4f}")
    print(f"📊 Final R² Score: {r2:.4f}")
    print(f"📊 Best Validation Loss: {best_val_loss:.4f}")
    
    # Log final metrics
    final_summary = {
        "final/train_loss": avg_train_loss,
        "final/val_loss": avg_val_loss,
        "final/r2_score": r2,
        "final/best_val_loss": best_val_loss,
        "final/total_epochs": config["epochs"],
        "final/model_parameters": sum(p.numel() for p in model.parameters())
    }
    
    wandb.log(final_summary)
    
    # Save final model as artifact
    model_path = "final_cpu_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_summary
    }, model_path)
    
    artifact = wandb.Artifact(
        name="cpu_allocation_final_model",
        type="model",
        description="Final trained CPU allocation model with config and metrics"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    # Clean up
    import os
    os.remove(model_path)
    
    print(f"\\n🌐 View complete results in wandb dashboard!")
    if hasattr(run, 'url') and run.url:
        print(f"🔗 Direct link: {run.url}")
    
    # Finish wandb run
    wandb.finish()
    
    return final_summary


def main():
    """Main function to run training with wandb web dashboard."""
    
    try:
        # Setup wandb for web dashboard
        setup_wandb_for_local_web()
        
        print("\\n🚀 Starting training with web dashboard logging...")
        print("💡 TIP: Open your browser and go to the wandb URL shown above")
        print("📊 You'll see real-time training metrics and visualizations!")
        
        # Run training
        results = train_with_web_logging()
        
        print("\\n🎉 Training completed successfully!")
        print("📊 Check your wandb dashboard for detailed analysis")
        
        # Show sync instructions if in offline mode
        if os.environ.get("WANDB_MODE") == "offline":
            print("\\n🔄 TO SYNC OFFLINE RUNS TO WEB DASHBOARD:")
            print("   1. Run: wandb online")
            print("   2. Run: wandb sync wandb/latest-run")
            print("   3. Visit: https://wandb.ai")
        
    except KeyboardInterrupt:
        print("\\n🛑 Training interrupted by user")
        wandb.finish()
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        wandb.finish()


if __name__ == "__main__":
    main()

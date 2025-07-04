"""
Wandb Local Web Dashboard Setup and Training
"""

import os
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np
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


def setup_wandb_local():
    """Setup wandb for local web dashboard."""
    
    print("🌐 Setting up Wandb Local Web Dashboard")
    print("=" * 50)
    
    # Set wandb to online mode for local dashboard
    os.environ["WANDB_MODE"] = "online"
    
    # Check if wandb is logged in
    try:
        # Try to get current user
        result = subprocess.run(['wandb', 'whoami'], capture_output=True, text=True)
        if result.returncode != 0:
            print("🔐 Logging into wandb...")
            # Create a local wandb account/project
            subprocess.run(['wandb', 'login', '--anonymously'], check=True)
        else:
            print(f"✅ Already logged in as: {result.stdout.strip()}")
    except Exception as e:
        print(f"⚠️  Wandb login issue: {e}")
        print("📱 Setting up anonymous mode...")
        os.environ["WANDB_ANONYMOUS"] = "allow"
    
    return True


def start_local_dashboard():
    """Start the local wandb dashboard in a separate thread."""
    
    def run_dashboard():
        try:
            print("🚀 Starting wandb local dashboard...")
            # Start wandb dashboard on localhost
            subprocess.run(['wandb', 'dashboard'], check=False)
        except Exception as e:
            print(f"⚠️  Dashboard startup issue: {e}")
    
    # Start dashboard in background thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Give it a moment to start
    time.sleep(3)
    
    # Try to open browser
    try:
        dashboard_url = "http://localhost:6006"
        print(f"🌐 Opening dashboard at: {dashboard_url}")
        webbrowser.open(dashboard_url)
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print("🖥️  Manually open: http://localhost:6006")


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


def train_with_local_wandb():
    """Train model with local wandb web dashboard."""
    
    # Configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 40,
        "model_name": "SimpleCPUModel_LocalDashboard",
        "hidden_dims": [128, 64, 32],
        "n_features": 10,
        "n_samples": 1200,
        "optimizer": "Adam",
        "loss_function": "MSE",
        "experiment_type": "local_dashboard_demo"
    }
    
    # Initialize wandb with local dashboard
    run = wandb.init(
        project="cpu-allocation-local-dashboard",
        name=f"local-experiment-{int(time.time())}",
        config=config,
        tags=["local-dashboard", "cpu-allocation", "demo"]
    )
    
    print("🚀 Starting CPU Allocation Model Training")
    print(f"📊 Wandb Run: {run.name}")
    print(f"🔗 Dashboard: {run.url}")
    print("=" * 60)
    
    # Create synthetic data
    print("📊 Creating synthetic CPU allocation data...")
    X, y = create_synthetic_data(
        n_samples=config["n_samples"], 
        n_features=config["n_features"]
    )
    
    # Log data statistics
    wandb.log({
        "data_samples": len(X),
        "data_features": X.shape[1],
        "target_mean": np.mean(y),
        "target_std": np.std(y),
        "target_min": np.min(y),
        "target_max": np.max(y)
    })
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X, y, batch_size=config["batch_size"]
    )
    
    # Initialize model
    print("🧠 Initializing model...")
    model = SimpleCPUModel(
        input_dim=config["n_features"],
        hidden_dims=config["hidden_dims"]
    )
    
    # Watch model with wandb
    wandb.watch(model, log_freq=50, log_graph=True)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"🔧 Device: {device}")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Log model info
    wandb.log({
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device)
    })
    
    # Training loop with detailed logging
    print("\\n🎯 Starting training with real-time dashboard updates...")
    
    best_val_loss = float('inf')
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Log batch-level metrics every 20 batches
            if batch_idx % 20 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                loss = criterion(output, target)
                mae = torch.mean(torch.abs(output - target))
                
                val_loss += loss.item()
                val_mae += mae.item()
                val_batches += 1
                
                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_mae = val_mae / val_batches
        
        # Calculate R²
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best model as wandb artifact
            torch.save(model.state_dict(), "best_model_temp.pt")
            
        # Log comprehensive epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mae": avg_val_mae,
            "val_r2": r2,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "best_val_loss": best_val_loss
        }
        
        wandb.log(epoch_metrics)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == config["epochs"] - 1:
            print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | R²: {r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Final model evaluation and logging
    print("\\n📊 Final evaluation and artifact creation...")
    
    # Create final visualizations
    import matplotlib.pyplot as plt
    
    # Prediction vs Actual plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(targets, predictions, alpha=0.6, color='blue')
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('Actual CPU Allocation (%)')
    plt.ylabel('Predicted CPU Allocation (%)')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = targets - predictions
    plt.scatter(predictions, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted CPU Allocation (%)')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Log final plot to wandb
    wandb.log({"final_predictions_analysis": wandb.Image(plt)})
    plt.close()
    
    # Create and log model artifact
    if Path("best_model_temp.pt").exists():
        artifact = wandb.Artifact(
            name="cpu_allocation_best_model",
            type="model",
            description="Best performing CPU allocation model from local training"
        )
        artifact.add_file("best_model_temp.pt")
        wandb.log_artifact(artifact)
        os.remove("best_model_temp.pt")
    
    # Log final summary metrics
    final_summary = {
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_val_mae": avg_val_mae,
        "final_r2_score": r2,
        "best_val_loss_achieved": best_val_loss,
        "total_epochs": config["epochs"],
        "total_parameters": sum(p.numel() for p in model.parameters())
    }
    
    wandb.log(final_summary)
    
    print("\\n✅ Training completed!")
    print(f"📊 Final R² Score: {r2:.4f}")
    print(f"📊 Final Validation Loss: {avg_val_loss:.4f}")
    print(f"📊 Best Validation Loss: {best_val_loss:.4f}")
    print(f"🌐 View detailed results in wandb dashboard: {run.url}")
    
    # Keep the run active for dashboard viewing
    print("\\n🖥️  Dashboard is running at: http://localhost:6006")
    print("💡 Press Ctrl+C to finish and close the wandb run")
    
    try:
        # Keep script running so dashboard stays active
        input("\\n⏳ Press Enter to finish the wandb run and exit...")
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
    
    # Finish wandb run
    wandb.finish()
    
    return model, final_summary


def main():
    """Main function to setup and run training with local wandb dashboard."""
    
    try:
        # Setup wandb for local dashboard
        setup_wandb_local()
        
        # Start local dashboard
        start_local_dashboard()
        
        # Wait a moment for dashboard to start
        time.sleep(2)
        
        # Run training with wandb logging
        model, results = train_with_local_wandb()
        
        print("\\n🎉 Training completed successfully!")
        print("📊 Check the wandb dashboard for detailed visualizations")
        
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

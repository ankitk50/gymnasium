"""
Simple Wandb Training Example for CPU Allocation Model
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
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


def train_model_with_wandb():
    """Train a CPU allocation model with Wandb logging."""
    
    # Set wandb mode (offline for demo)
    os.environ["WANDB_MODE"] = "offline"
    
    # Configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,  # Reduced for faster demo
        "model_name": "SimpleCPUModel",
        "hidden_dims": [64, 32],
        "n_features": 10,
        "n_samples": 1000,
        "optimizer": "Adam",
        "loss_function": "MSE"
    }
    
    # Initialize wandb
    wandb.init(
        project="cpu-allocation-demo",
        name="simple-cpu-model-experiment",
        config=config
    )
    
    print("🚀 Starting CPU Allocation Model Training with Wandb")
    print("=" * 60)
    
    # Create synthetic data
    print("📊 Creating synthetic CPU allocation data...")
    X, y = create_synthetic_data(
        n_samples=config["n_samples"], 
        n_features=config["n_features"]
    )
    
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
    wandb.watch(model, log_freq=100)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print(f"🔧 Device: {device}")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\\n🎯 Starting training...")
    
    for epoch in range(config["epochs"]):
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
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Final evaluation
    print("\\n📊 Final evaluation...")
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions.extend(output.cpu().numpy().flatten())
            actuals.extend(target.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate final metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    
    print(f"📊 Final MSE: {mse:.4f}")
    print(f"📊 Final MAE: {mae:.4f}")
    print(f"📊 Final R²: {r2:.4f}")
    
    # Log final metrics
    wandb.log({
        "final_mse": mse,
        "final_mae": mae,
        "final_r2": r2
    })
    
    # Create prediction vs actual plot
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual CPU Allocation (%)')
    plt.ylabel('Predicted CPU Allocation (%)')
    plt.title('Predictions vs Actual')
    
    plt.subplot(1, 2, 2)
    residuals = actuals - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted CPU Allocation (%)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    
    # Log plot to wandb
    wandb.log({"prediction_analysis": wandb.Image(plt)})
    plt.close()
    
    # Save model as artifact
    model_path = "cpu_allocation_model.pt"
    torch.save(model.state_dict(), model_path)
    
    artifact = wandb.Artifact(
        name="cpu_allocation_model",
        type="model",
        description="Trained CPU allocation prediction model"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    print("\\n✅ Training completed!")
    print(f"📁 Model saved as artifact")
    print(f"🌐 Wandb run completed")
    
    # Clean up
    os.remove(model_path)
    
    # Finish wandb run
    wandb.finish()
    
    return model, mse, mae, r2


if __name__ == "__main__":
    try:
        model, mse, mae, r2 = train_model_with_wandb()
        print("\\n🎉 Success! Training completed with wandb integration.")
        print(f"📊 Final Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        wandb.finish()

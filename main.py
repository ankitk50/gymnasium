"""
Main entry point for the ML/DL training pipeline framework.
Includes Wandb localhost configuration and experiment management.
"""

import argparse
import sys
import os
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.pipeline import TrainingPipeline
from src.utils.config import Config, create_default_config
from src.utils.logging import setup_logging


def check_wandb_installation():
    """Check if wandb is properly installed."""
    try:
        result = subprocess.run(['wandb', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Wandb installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Wandb not found")
            return False
    except FileNotFoundError:
        print("❌ Wandb command not found")
        return False


def setup_wandb_localhost():
    """Setup wandb for localhost operation."""
    print("🌐 CONFIGURING WANDB FOR LOCALHOST")
    print("=" * 50)
    
    # Check installation
    if not check_wandb_installation():
        print("📦 Installing wandb...")
        subprocess.run(['pip', 'install', 'wandb'], check=True)
    
    # Configure environment for localhost
    localhost_configs = {
        "WANDB_MODE": "online",
        "WANDB_API_KEY": "local-44761a1fe98b19207436e87edcb7a9824731aa01",
        "WANDB_BASE_URL": "http://localhost:8080",
    }
    
    for key, value in localhost_configs.items():
        os.environ[key] = value
        print(f"🔧 Set {key}={value}")
    
    try:
        print("🔐 Setting up wandb login...")
        result = subprocess.run(['wandb', 'login', '--anonymously'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Anonymous login successful")
        else:
            print("⚠️  Using offline mode as fallback")
            os.environ["WANDB_MODE"] = "offline"
    except Exception as e:
        print(f"⚠️  Login issue: {e}. Using offline mode")
        os.environ["WANDB_MODE"] = "offline"
    
    print("✅ Wandb localhost configuration complete!")
    return True


def start_wandb_server(background=True):
    """Start wandb local server."""
    def run_server():
        try:
            print("🚀 Starting wandb local server...")
            print("🌐 Server will be available at: http://localhost:8080")
            
            process = subprocess.Popen(
                ['wandb', 'server', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line.strip():
                        print(f"📡 Server: {line.strip()}")
                        if "started" in line.lower() or "running" in line.lower():
                            break
            
        except Exception as e:
            print(f"❌ Server startup error: {e}")
    
    if background:
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)  # Give server time to start
        return server_thread
    else:
        run_server()
        return None


def open_wandb_dashboard():
    """Open wandb dashboard in browser."""
    dashboard_urls = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:6006"
    ]
    
    for url in dashboard_urls:
        try:
            print(f"🌐 Opening dashboard at: {url}")
            webbrowser.open(url)
            print(f"✅ Dashboard opened at: {url}")
            return url
        except Exception as e:
            print(f"⚠️  Could not open {url}: {e}")
    
    print("🖥️  Please manually open one of these URLs:")
    for url in dashboard_urls:
        print(f"   {url}")
    return None


def sync_offline_runs():
    """Sync any existing offline wandb runs."""
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        offline_runs = [d for d in wandb_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("offline-run")]
        
        if offline_runs:
            print(f"🔄 Found {len(offline_runs)} offline runs to sync...")
            
            for run_dir in offline_runs:
                try:
                    print(f"📤 Syncing: {run_dir.name}")
                    result = subprocess.run(
                        ['wandb', 'sync', str(run_dir)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ Synced: {run_dir.name}")
                    else:
                        print(f"⚠️  Sync failed for {run_dir.name}")
                        
                except Exception as e:
                    print(f"❌ Error syncing {run_dir.name}: {e}")
        else:
            print("📂 No offline runs found to sync")
    else:
        print("📂 No wandb directory found")


def show_wandb_instructions():
    """Show instructions for using wandb dashboard."""
    instructions = """
    🎯 WANDB LOCAL DASHBOARD INSTRUCTIONS
    ====================================
    
    🚀 WHAT'S RUNNING:
    • Wandb local server at http://localhost:8080
    • Real-time experiment tracking
    • Interactive visualizations
    
    📊 DASHBOARD FEATURES:
    • Training/validation loss curves
    • Learning rate schedules  
    • Model gradients and weights
    • Custom metrics and plots
    • Model artifacts and checkpoints
    • Hyperparameter comparisons
    
    💡 TIPS:
    • Dashboard updates in real-time during training
    • Click on runs to see detailed metrics
    • Use the web interface to compare experiments
    • All data is stored locally
    """
    print(instructions)


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='ML/DL Training Pipeline Framework with Wandb Integration')
    
    # Experiment configuration
    parser.add_argument('--config', '-c', type=str, default='configs/cpu_allocation.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', '-e', type=str, default=None,
                       help='Experiment name (overrides config)')
    
    # Training parameters
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # Execution modes
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation (requires existing model)')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Run cross-validation')
    parser.add_argument('--hyperparameter-search', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--model-checkpoint', type=str, default=None,
                       help='Path to model checkpoint to load')
    
    # Wandb configuration
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='gymnasium-ml-pipeline',
                       help='Wandb project name')
    parser.add_argument('--setup-wandb-localhost', action='store_true', default=True,
                       help='Setup and configure Wandb for localhost operation')
    parser.add_argument('--start-wandb-server', action='store_true',
                       help='Start Wandb local server')
    parser.add_argument('--open-dashboard', action='store_true',
                       help='Open Wandb dashboard in browser')
    parser.add_argument('--sync-offline', action='store_true',
                       help='Sync offline Wandb runs')
    parser.add_argument('--wandb-server-only', action='store_true',
                       help='Only start Wandb server (no training)')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level=log_level)
    
    # Handle Wandb setup and server operations first
    if args.setup_wandb_localhost or args.start_wandb_server or args.wandb_server_only:
        print("🌐 WANDB LOCALHOST SETUP")
        print("=" * 50)
        
        # Setup wandb localhost configuration
        if args.setup_wandb_localhost or args.start_wandb_server or args.wandb_server_only:
            setup_wandb_localhost()
        
        # Sync offline runs if requested
        if args.sync_offline:
            sync_offline_runs()
        
        # Start server
        server_thread = None
        if args.start_wandb_server or args.wandb_server_only:
            server_thread = start_wandb_server(background=(not args.wandb_server_only))
            time.sleep(2)  # Give server time to start
        
        # Open dashboard
        if args.open_dashboard:
            open_wandb_dashboard()
        
        # If only server mode, keep running
        if args.wandb_server_only:
            show_wandb_instructions()
            print("\n⏳ Wandb server is running... Press Ctrl+C to stop")
            print("🌐 Dashboard URL: http://localhost:8080")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping wandb server...")
                try:
                    subprocess.run(['wandb', 'server', 'stop'], timeout=10)
                    print("✅ Server stopped")
                except:
                    print("⚠️  Server may still be running")
                return
        
        print("✅ Wandb setup complete! Proceeding with experiment...\n")
        
        # Enable wandb by default when using localhost setup
        if not args.use_wandb:
            args.use_wandb = True
            print("🔧 Auto-enabled --use-wandb for localhost setup")
    
    try:
        # Load configuration
        if Path(args.config).exists():
            config = Config.from_yaml(args.config)
        else:
            print(f"Config file {args.config} not found. Using default configuration.")
            config = create_default_config()
        
        # Apply command line overrides
        if args.experiment_name:
            config.set('experiment_name', args.experiment_name)
        if args.device:
            config.set('device', args.device)
        if args.epochs:
            config.set('training.epochs', args.epochs)
        if args.batch_size:
            config.set('data.batch_size', args.batch_size)
        if args.learning_rate:
            config.set('model.learning_rate', args.learning_rate)
        
        # Initialize pipeline
        pipeline = TrainingPipeline(
            config_dict=config.to_dict(),
            experiment_name=config.get('experiment_name'),
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project
        )
        
        print(f"Starting experiment: {config.get('experiment_name')}")
        print(f"Device: {config.get('device')}")
        print(f"Model: {config.get('model.name')}")
        
        # Load checkpoint if specified
        if args.model_checkpoint:
            print(f"Loading model checkpoint: {args.model_checkpoint}")
            pipeline.load_checkpoint(args.model_checkpoint)
        
        # Run based on mode
        if args.evaluate_only:
            # Evaluation only
            print("Running evaluation...")
            pipeline.setup_data()
            if not args.model_checkpoint:
                raise ValueError("Model checkpoint required for evaluation-only mode")
            results = pipeline.evaluate()
            print("Evaluation completed!")
            
        elif args.cross_validate:
            # Cross-validation
            print("Running cross-validation...")
            cv_results = pipeline.cross_validate(k_folds=5)
            print("Cross-validation completed!")
            print("CV Results:", cv_results['cv_metrics'])
            
        elif args.hyperparameter_search:
            # Hyperparameter optimization
            print("Running hyperparameter search...")
            
            # Define search space (example)
            search_space = {
                'model.learning_rate': {'type': 'float', 'range': [1e-5, 1e-2]},
                'model.hidden_dims.0': {'type': 'int', 'range': [64, 256]},
                'model.dropout_rate': {'type': 'float', 'range': [0.1, 0.5]},
                'training.batch_size': {'type': 'categorical', 'range': [16, 32, 64]}
            }
            
            hp_results = pipeline.hyperparameter_search(search_space, n_trials=20)
            print("Hyperparameter search completed!")
            print("Best parameters:", hp_results['best_params'])
            print("Best value:", hp_results['best_value'])
            
        else:
            # Normal training and evaluation
            print("Starting training...")
            trained_model = pipeline.train()
            
            print("Training completed! Starting evaluation...")
            results = pipeline.evaluate(trained_model)
            
            print(f"Pipeline completed successfully!")
            print(f"Results saved to: {pipeline.output_dir}")
            
            # Print key metrics
            metrics = results.get('metrics', {})
            if 'mse' in metrics:
                print(f"Final MSE: {metrics['mse']:.4f}")
            if 'mae' in metrics:
                print(f"Final MAE: {metrics['mae']:.4f}")
            if 'r2_score' in metrics:
                print(f"Final R²: {metrics['r2_score']:.4f}")
            
            # Show wandb dashboard info if using wandb
            if args.use_wandb:
                print("\n🌐 WANDB DASHBOARD")
                print("=" * 30)
                if 'WANDB_BASE_URL' in os.environ:
                    print(f"📊 View results at: {os.environ['WANDB_BASE_URL']}")
                else:
                    print("📊 View results at: https://wandb.ai")
                print("💡 Dashboard shows real-time training metrics and model performance")
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

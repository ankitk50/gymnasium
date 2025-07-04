"""
Setup and Run Wandb Local Server for Web Logging
"""

import os
import subprocess
import time
import webbrowser
import threading
from pathlib import Path


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


def setup_wandb_local_server():
    """Setup wandb for local web server."""
    
    print("🌐 SETTING UP WANDB LOCAL WEB DASHBOARD")
    print("=" * 50)
    
    # Check installation
    if not check_wandb_installation():
        print("📦 Installing wandb...")
        subprocess.run(['pip', 'install', 'wandb'], check=True)
    
    # Set environment for local usage
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_API_KEY"] = "local"
    
    # Try to login anonymously for local use
    try:
        print("🔐 Setting up anonymous wandb access...")
        result = subprocess.run(['wandb', 'login', '--anonymously'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Anonymous login successful")
        else:
            print("⚠️  Using offline mode instead")
            os.environ["WANDB_MODE"] = "offline"
    except Exception as e:
        print(f"⚠️  Login issue: {e}")
        print("📱 Using offline mode")
        os.environ["WANDB_MODE"] = "offline"


def start_wandb_server():
    """Start the wandb local server."""
    
    def run_server():
        try:
            print("🚀 Starting wandb local server...")
            print("🌐 Server will be available at: http://localhost:6006")
            
            # Start wandb server
            process = subprocess.Popen(
                ['wandb', 'server', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor output
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line.strip():
                        print(f"📡 Server: {line.strip()}")
                        if "started" in line.lower() or "running" in line.lower():
                            break
            
        except Exception as e:
            print(f"❌ Server startup error: {e}")
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(5)
    
    return server_thread


def open_dashboard():
    """Open the wandb dashboard in browser."""
    
    dashboard_urls = [
        "http://localhost:6006",
        "http://127.0.0.1:6006",
        "http://localhost:8080"  # Alternative port
    ]
    
    for url in dashboard_urls:
        try:
            print(f"🌐 Attempting to open: {url}")
            webbrowser.open(url)
            print(f"✅ Dashboard opened at: {url}")
            return url
        except Exception as e:
            print(f"⚠️  Could not open {url}: {e}")
    
    print("🖥️  Please manually open one of these URLs:")
    for url in dashboard_urls:
        print(f"   {url}")


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
                        print(f"⚠️  Sync failed for {run_dir.name}: {result.stderr}")
                        
                except Exception as e:
                    print(f"❌ Error syncing {run_dir.name}: {e}")
        else:
            print("📂 No offline runs found to sync")
    else:
        print("📂 No wandb directory found")


def show_dashboard_instructions():
    """Show instructions for using the wandb dashboard."""
    
    instructions = """
    🎯 WANDB LOCAL DASHBOARD INSTRUCTIONS
    ====================================
    
    🚀 WHAT'S RUNNING:
    • Wandb local server at http://localhost:6006
    • Real-time experiment tracking
    • Interactive visualizations
    
    📊 DASHBOARD FEATURES:
    • Training/validation loss curves
    • Learning rate schedules  
    • Model gradients and weights
    • Custom metrics and plots
    • Model artifacts and checkpoints
    • Hyperparameter comparisons
    
    🔧 TO START TRAINING:
    1. Keep this terminal open (server running)
    2. In a new terminal, run your training script
    3. Visit http://localhost:6006 to see live updates
    
    📈 EXAMPLE TRAINING COMMAND:
    python simple_wandb_training.py
    
    💡 TIPS:
    • Refresh the dashboard to see new experiments
    • Use wandb.log() in your code for custom metrics
    • Dashboard updates in real-time during training
    • Click on runs to see detailed metrics
    
    🛑 TO STOP:
    • Press Ctrl+C in this terminal
    • Or close the browser tab
    """
    
    print(instructions)


def main():
    """Main function to setup wandb local web logging."""
    
    try:
        # Setup wandb
        setup_wandb_local_server()
        
        # Sync existing offline runs
        sync_offline_runs()
        
        # Start the server
        server_thread = start_wandb_server()
        
        # Open dashboard
        dashboard_url = open_dashboard()
        
        # Show instructions
        show_dashboard_instructions()
        
        print("\\n⏳ Server is running... Press Ctrl+C to stop")
        print("🌐 Dashboard URL: http://localhost:6006")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 Stopping wandb server...")
            
            # Try to stop wandb server gracefully
            try:
                subprocess.run(['wandb', 'server', 'stop'], timeout=10)
                print("✅ Server stopped")
            except:
                print("⚠️  Server may still be running")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

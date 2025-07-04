"""
Complete Guide: Wandb Local Web Dashboard for Model Training
"""

import os
import subprocess
import webbrowser
import time
from pathlib import Path


def show_wandb_setup_guide():
    """Show complete guide for setting up wandb local web dashboard."""
    
    guide = """
    🌐 WANDB LOCAL WEB DASHBOARD SETUP GUIDE
    ========================================
    
    📋 OVERVIEW:
    This guide shows you how to set up Weights & Biases (wandb) for local 
    web dashboard viewing of your machine learning experiments.
    
    🎯 WHAT YOU'LL GET:
    • Real-time training metrics in your browser
    • Interactive loss curves and visualizations  
    • Model architecture graphs
    • Hyperparameter tracking
    • Model artifacts and checkpoints
    • Experiment comparison tools
    
    🚀 SETUP METHODS:
    
    METHOD 1: ONLINE DASHBOARD (Recommended)
    ========================================
    1. Sign up at https://wandb.ai (free account)
    2. Login: wandb login
    3. Use WANDB_MODE="online" in your code
    4. View at: https://wandb.ai/your-username/project-name
    
    METHOD 2: LOCAL OFFLINE + SYNC
    ==============================
    1. Use WANDB_MODE="offline" in your code
    2. Train your model (creates local logs)
    3. Sync later: wandb sync wandb/offline-run-*
    4. View at: https://wandb.ai
    
    METHOD 3: PURE LOCAL (No Internet)
    ==================================
    1. Use WANDB_MODE="offline" 
    2. Train your model
    3. View logs locally with custom scripts
    4. Or export to other visualization tools
    
    📊 INTEGRATION IN YOUR CODE:
    ============================
    """
    
    code_example = '''
    import wandb
    import torch
    
    # Initialize wandb
    wandb.init(
        project="my-ml-project",
        name="experiment-1",
        config={"lr": 0.001, "epochs": 50}
    )
    
    # Watch your model
    model = MyModel()
    wandb.watch(model)
    
    # Training loop
    for epoch in range(epochs):
        # ... training code ...
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        })
        
        # Log images/plots
        wandb.log({"predictions": wandb.Image(plt)})
    
    # Finish run
    wandb.finish()
    '''
    
    print(guide)
    print("CODE EXAMPLE:")
    print("=" * 50)
    print(code_example)


def check_wandb_status():
    """Check current wandb installation and login status."""
    
    print("🔍 CHECKING WANDB STATUS")
    print("=" * 30)
    
    # Check installation
    try:
        result = subprocess.run(['wandb', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Wandb installed: {result.stdout.strip()}")
        else:
            print("❌ Wandb not properly installed")
            return False
    except FileNotFoundError:
        print("❌ Wandb not found. Install with: pip install wandb")
        return False
    
    # Check login status
    try:
        result = subprocess.run(['wandb', 'whoami'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Logged in as: {result.stdout.strip()}")
        else:
            print("⚠️  Not logged in. Run: wandb login")
    except:
        print("⚠️  Could not check login status")
    
    # Check for existing runs
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        runs = list(wandb_dir.glob("*run*"))
        print(f"📁 Found {len(runs)} wandb runs in current directory")
        
        if runs:
            print("   Recent runs:")
            for run in sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                print(f"   • {run.name}")
    else:
        print("📁 No wandb runs found in current directory")
    
    return True


def sync_offline_runs():
    """Sync offline wandb runs to web dashboard."""
    
    print("\\n🔄 SYNCING OFFLINE RUNS")
    print("=" * 30)
    
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        print("📁 No wandb directory found")
        return
    
    offline_runs = [d for d in wandb_dir.iterdir() 
                   if d.is_dir() and "offline-run" in d.name]
    
    if not offline_runs:
        print("📁 No offline runs found")
        return
    
    print(f"📤 Found {len(offline_runs)} offline runs to sync")
    
    for i, run_dir in enumerate(offline_runs, 1):
        print(f"\\n🔄 Syncing run {i}/{len(offline_runs)}: {run_dir.name}")
        
        try:
            result = subprocess.run(
                ['wandb', 'sync', str(run_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully synced: {run_dir.name}")
                if "View run at" in result.stdout:
                    # Extract URL from output
                    lines = result.stdout.split('\\n')
                    for line in lines:
                        if "https://" in line:
                            print(f"🔗 Dashboard: {line.strip()}")
                            break
            else:
                print(f"❌ Failed to sync: {run_dir.name}")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout syncing: {run_dir.name}")
        except Exception as e:
            print(f"❌ Error syncing {run_dir.name}: {e}")


def open_wandb_dashboard():
    """Open wandb dashboard in browser."""
    
    print("\\n🌐 OPENING WANDB DASHBOARD")
    print("=" * 30)
    
    # Try to get user info and open their dashboard
    try:
        result = subprocess.run(['wandb', 'whoami'], capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            dashboard_url = f"https://wandb.ai/{username}"
            print(f"🔗 Opening dashboard: {dashboard_url}")
            webbrowser.open(dashboard_url)
            return dashboard_url
        else:
            print("⚠️  Not logged in. Opening general wandb site...")
            webbrowser.open("https://wandb.ai")
            return "https://wandb.ai"
    except Exception as e:
        print(f"❌ Error opening dashboard: {e}")
        print("🖥️  Manually visit: https://wandb.ai")
        return None


def show_wandb_commands():
    """Show useful wandb commands."""
    
    commands = """
    🛠️  USEFUL WANDB COMMANDS
    =========================
    
    📊 Basic Commands:
    • wandb login                 - Login to wandb
    • wandb logout                - Logout from wandb  
    • wandb whoami                - Check current user
    • wandb status                - Check wandb status
    
    🔄 Sync Commands:
    • wandb sync wandb/offline-run-*    - Sync specific run
    • wandb sync wandb/                 - Sync all offline runs
    • wandb online                      - Switch to online mode
    • wandb offline                     - Switch to offline mode
    
    📁 Project Commands:
    • wandb projects                    - List your projects
    • wandb runs PROJECT_NAME           - List runs in project
    • wandb sweep config.yaml           - Start hyperparameter sweep
    
    🔧 Debug Commands:
    • wandb doctor                      - Diagnose issues
    • wandb verify                      - Verify installation
    • wandb disabled                    - Disable wandb temporarily
    
    🌐 Dashboard Commands:
    • Open: https://wandb.ai
    • Or: wandb dashboard (if running local server)
    """
    
    print(commands)


def main():
    """Main function to guide user through wandb setup."""
    
    print("🎯 WANDB LOCAL WEB DASHBOARD GUIDE")
    print("=" * 50)
    
    # Show setup guide
    show_wandb_setup_guide()
    
    # Check current status
    if not check_wandb_status():
        print("\\n❌ Please install wandb first: pip install wandb")
        return
    
    # Sync existing offline runs
    sync_offline_runs()
    
    # Show useful commands
    show_wandb_commands()
    
    # Offer to open dashboard
    print("\\n" + "=" * 50)
    try:
        response = input("🌐 Would you like to open the wandb dashboard? (y/n): ").lower()
        if response in ['y', 'yes']:
            open_wandb_dashboard()
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    
    # Final tips
    final_tips = """
    💡 NEXT STEPS:
    ==============
    
    1. 🔐 Login to wandb: wandb login
    2. 🏃 Run your training script with wandb.init()
    3. 📊 Watch real-time metrics in your browser
    4. 🔄 Sync offline runs when online: wandb sync wandb/
    5. 📈 Compare experiments in the dashboard
    
    🚀 Ready to start training with wandb web logging!
    """
    
    print(final_tips)


if __name__ == "__main__":
    main()

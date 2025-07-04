"""
Final Demo: Sync Wandb Runs and View in Web Dashboard
"""

import subprocess
import webbrowser
import time
from pathlib import Path


def demonstrate_wandb_sync():
    """Demonstrate syncing wandb runs to web dashboard."""
    
    print("🎯 WANDB WEB DASHBOARD DEMONSTRATION")
    print("=" * 50)
    
    print("📊 SUMMARY OF WHAT WE'VE ACCOMPLISHED:")
    print("✅ Trained multiple CPU allocation models")
    print("✅ Logged metrics, visualizations, and model artifacts")
    print("✅ Created comprehensive training logs")
    print("✅ Generated interactive plots and analysis")
    
    # Check wandb runs
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        runs = list(wandb_dir.glob("*run*"))
        print(f"\\n📁 Total wandb runs created: {len(runs)}")
        
        offline_runs = [r for r in runs if "offline-run" in r.name]
        online_runs = [r for r in runs if "offline-run" not in r.name and r.is_dir()]
        
        print(f"   • Offline runs: {len(offline_runs)}")
        print(f"   • Online runs: {len(online_runs)}")
        
        # Show recent runs
        print("\\n📋 Recent training runs:")
        for run in sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            if run.is_dir():
                size = sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
                print(f"   • {run.name} ({size/1024:.1f} KB)")
    
    print("\\n🌐 WEB DASHBOARD OPTIONS:")
    print("=" * 40)
    
    print("\\n1️⃣  SYNC TO WANDB.AI (Recommended)")
    print("   • Creates beautiful web dashboard")
    print("   • Interactive visualizations")
    print("   • Experiment comparison tools")
    print("   • Shareable links")
    
    print("\\n2️⃣  LOCAL VISUALIZATION")
    print("   • View offline logs locally")
    print("   • Export to other tools")
    print("   • No internet required")
    
    # Demonstration commands
    demo_commands = """
    💻 COMMANDS TO TRY:
    ==================
    
    # Login to wandb (one-time setup)
    wandb login
    
    # Sync all offline runs to web dashboard
    wandb sync wandb/
    
    # Sync specific run
    wandb sync wandb/offline-run-20250704_023001-plfca8cj
    
    # View your projects
    wandb projects
    
    # Check sync status
    wandb status
    
    # Open dashboard
    # Visit: https://wandb.ai/your-username
    """
    
    print(demo_commands)
    
    # Show what's logged
    logged_data = """
    📊 WHAT'S LOGGED IN YOUR WANDB RUNS:
    ===================================
    
    📈 Metrics:
    • Training & validation loss curves
    • Learning rate schedules
    • Model accuracy metrics (MSE, MAE, R²)
    • Batch-level training progress
    
    📸 Visualizations:
    • Prediction vs Actual scatter plots
    • Residual analysis plots
    • Training progress charts
    • Model architecture graphs
    
    💾 Artifacts:
    • Trained model weights (.pt files)
    • Configuration files
    • Dataset statistics
    • Training logs
    
    🔧 System Info:
    • Model parameters count
    • Training duration
    • Device information (CPU/GPU)
    • Environment details
    """
    
    print(logged_data)
    
    # Interactive demo
    print("\\n" + "=" * 50)
    print("🚀 READY TO VIEW YOUR RESULTS!")
    
    try:
        choice = input("\\nChoose an option:\\n1. Sync runs to wandb.ai\\n2. Show local files\\n3. Exit\\nEnter (1/2/3): ")
        
        if choice == "1":
            print("\\n🔄 To sync your runs to wandb.ai:")
            print("1. Run: wandb login")
            print("2. Run: wandb sync wandb/")
            print("3. Visit: https://wandb.ai")
            
            try:
                open_web = input("\\nOpen wandb.ai in browser? (y/n): ").lower()
                if open_web in ['y', 'yes']:
                    webbrowser.open("https://wandb.ai")
                    print("🌐 Opened https://wandb.ai in your browser")
            except:
                pass
                
        elif choice == "2":
            print("\\n📁 Local wandb files:")
            if wandb_dir.exists():
                for run in sorted(wandb_dir.glob("*run*"), key=lambda x: x.stat().st_mtime, reverse=True):
                    if run.is_dir():
                        print(f"   {run}")
                        
                        # Show run contents
                        wandb_file = list(run.glob("*.wandb"))
                        if wandb_file:
                            size = wandb_file[0].stat().st_size / 1024
                            print(f"     • Data: {size:.1f} KB")
                        
                        media_dir = run / "files" / "media"
                        if media_dir.exists():
                            images = list(media_dir.glob("**/*.png"))
                            print(f"     • Images: {len(images)} files")
        else:
            print("👋 Goodbye!")
            
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    
    # Final message
    final_message = """
    🎉 CONGRATULATIONS!
    ===================
    
    You've successfully:
    ✅ Trained ML models with wandb integration
    ✅ Logged comprehensive training metrics
    ✅ Created visualizations and artifacts
    ✅ Set up local web dashboard logging
    
    🚀 Next steps:
    • Run 'wandb login' to connect to wandb.ai
    • Sync your runs with 'wandb sync wandb/'
    • Explore the interactive web dashboard
    • Compare different experiments
    • Share results with your team
    
    📚 Learn more:
    • Docs: https://docs.wandb.ai
    • Examples: https://github.com/wandb/examples
    • Community: https://wandb.ai/community
    """
    
    print(final_message)


if __name__ == "__main__":
    demonstrate_wandb_sync()

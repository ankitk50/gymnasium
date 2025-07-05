# Wandb Localhost Integration

This document describes the enhanced main.py script that includes Wandb localhost configuration and experiment management.

## Quick Start

### Option 1: One Command Setup & Run
```bash
python run_experiment_with_wandb.py
```

### Option 2: Manual Control
```bash
# Setup Wandb localhost and run experiment
python main.py --setup-wandb-localhost --start-wandb-server --open-dashboard --use-wandb

# Or just start the server
python main.py --wandb-server-only
```

### Option 3: Interactive Demo
```bash
python demo.py
```

## New Features in main.py

### Wandb Configuration Options
- `--setup-wandb-localhost`: Configure Wandb for localhost operation
- `--start-wandb-server`: Start local Wandb server
- `--open-dashboard`: Open dashboard in browser
- `--sync-offline`: Sync existing offline runs
- `--wandb-server-only`: Only start server (no training)

### Enhanced Experiment Management
- Automatic Wandb setup when using localhost options
- Real-time dashboard integration
- Offline run synchronization
- Background server management

## Usage Examples

### Basic Training with Wandb
```bash
python main.py --use-wandb --config configs/cpu_allocation.yaml
```

### Complete Setup + Training
```bash
python main.py \
    --setup-wandb-localhost \
    --start-wandb-server \
    --open-dashboard \
    --use-wandb \
    --experiment-name "my_experiment" \
    --epochs 50
```

### Hyperparameter Search
```bash
python main.py \
    --setup-wandb-localhost \
    --start-wandb-server \
    --use-wandb \
    --hyperparameter-search \
    --experiment-name "hp_search"
```

### Cross Validation
```bash
python main.py \
    --setup-wandb-localhost \
    --start-wandb-server \
    --use-wandb \
    --cross-validate \
    --experiment-name "cv_experiment"
```

## Dashboard Features

Access the dashboard at: http://localhost:8080

Features include:
- Real-time training/validation metrics
- Learning rate schedules
- Model gradients and weights
- Custom plots and visualizations
- Experiment comparison
- Model artifacts and checkpoints

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `main.py` | Enhanced main script with Wandb integration |
| `run_experiment_with_wandb.py` | Simplified one-command setup & run |
| `demo.py` | Interactive demo of the functionality |
| `usage_guide.py` | Comprehensive usage examples |

## Environment Variables

The localhost setup automatically configures:
```bash
WANDB_MODE=online
WANDB_API_KEY=local-44761a1fe98b19207436e87edcb7a9824731aa01
WANDB_BASE_URL=http://localhost:8080
```

## Troubleshooting

### Server Not Starting
```bash
# Kill existing processes
pkill -f "wandb server"

# Restart
python main.py --wandb-server-only
```

### Dashboard Not Opening
- Manually navigate to: http://localhost:8080
- Alternative ports: http://localhost:6006

### Connection Issues
```bash
# Check if server is running
ps aux | grep wandb

# Restart server
python main.py --start-wandb-server --wandb-server-only
```

## Integration with Existing Code

The enhanced main.py maintains full backward compatibility. All existing command-line options continue to work as before. The new Wandb features are additive and optional.

To use Wandb in your experiments, simply add `--use-wandb` to any existing command:
```bash
# Before
python main.py --config configs/cpu_allocation.yaml --epochs 100

# After (with local Wandb)
python main.py --setup-wandb-localhost --start-wandb-server --use-wandb --config configs/cpu_allocation.yaml --epochs 100
```

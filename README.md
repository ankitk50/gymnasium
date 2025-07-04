# ML/DL Training Pipeline Framework

A generic, reusable training pipeline framework for machine learning and deep learning models with built-in visualization and evaluation capabilities.

## Features

- **Generic Training Pipeline**: Modular framework supporting both traditional ML and deep learning models
- **CPU Allocation Model**: Specialized implementation for CPU allocation tasks
- **Comprehensive Visualization**: Training metrics, model performance, and inference visualization
- **Evaluation Suite**: Model assessment with multiple metrics and validation techniques
- **Configuration Management**: YAML-based configuration with Hydra integration
- **Hyperparameter Optimization**: Built-in support for Optuna
- **Experiment Tracking**: Integration with TensorBoard and Weights & Biases

## Project Structure

```
gymnasium/
├── src/                        # Core framework source code
│   ├── core/                   # Core framework components
│   │   ├── pipeline.py         # Main training pipeline
│   │   ├── base_model.py       # Base model interface
│   │   ├── trainer.py          # Training logic
│   │   └── evaluator.py        # Evaluation logic
│   ├── models/                 # Model implementations
│   │   ├── cpu_allocation.py   # CPU allocation specific models
│   │   └── registry.py         # Model registry
│   ├── data/                   # Data handling
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── preprocessor.py     # Data preprocessing
│   ├── visualization/          # Visualization components
│   │   ├── training_viz.py     # Training visualization
│   │   ├── evaluation_viz.py   # Evaluation visualization
│   │   └── inference_viz.py    # Inference visualization
│   └── utils/                  # Utility functions
│       ├── config.py           # Configuration management
│       ├── metrics.py          # Custom metrics
│       └── logging.py          # Logging utilities
├── configs/                    # Configuration files
├── docs/                       # Documentation
│   ├── DEMO_SUMMARY.md         # Demo overview
│   ├── QUICKSTART.md           # Quick start guide
│   └── WANDB_SETUP_COMPLETE.md # WandB setup guide
├── wandb_integration/          # Weights & Biases integration
│   ├── setup_wandb_server.py   # WandB server setup
│   ├── train_with_wandb.py     # Training with WandB
│   └── ...                     # Other WandB utilities
├── notebooks/                  # Jupyter notebooks for analysis
├── experiments/                # Experiment outputs (gitignored)
├── logs/                       # Training logs (gitignored)
├── results/                    # Results data (gitignored)
└── wandb/                      # WandB data (gitignored)
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run CPU allocation training:
```bash
python src/main.py --config-name=cpu_allocation
```

3. View results in TensorBoard:
```bash
tensorboard --logdir experiments/
```

For more detailed setup instructions, see `docs/QUICKSTART.md`.

## Documentation

Complete documentation is available in the `docs/` folder:
- `docs/QUICKSTART.md` - Detailed setup and usage guide
- `docs/DEMO_SUMMARY.md` - Overview of demo features
- `docs/WANDB_SETUP_COMPLETE.md` - Weights & Biases integration guide

## Weights & Biases Integration

WandB integration code is organized in the `wandb_integration/` folder. This includes:
- Training scripts with WandB logging
- Visualization utilities
- Server setup and configuration
- Demo examples

See `wandb_integration/README.md` for detailed information.

## Usage

### Training a Model

```python
from src.core.pipeline import TrainingPipeline
from src.models.cpu_allocation import CPUAllocationModel

# Initialize pipeline
pipeline = TrainingPipeline(config_path="configs/cpu_allocation.yaml")

# Train model
trained_model = pipeline.train()

# Evaluate model
results = pipeline.evaluate(trained_model)
```

### Custom Model Implementation

```python
from src.core.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your model implementation
    
    def forward(self, x):
        # Forward pass implementation
        pass
```

## Configuration

Models and training parameters are configured via YAML files in the `configs/` directory. See `configs/cpu_allocation.yaml` for an example.

## License

MIT License

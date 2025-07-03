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
├── src/
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
├── data/                       # Dataset storage
├── experiments/                # Experiment outputs
└── notebooks/                  # Jupyter notebooks for analysis
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

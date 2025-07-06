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
- **🔮 Advanced Inference Engine**: Production-ready inference with monitoring, validation, and drift detection
- **🔍 Model Validation Framework**: Comprehensive validation pipelines with robustness testing
- **🚀 Deployment Manager**: Integrated deployment solution with health monitoring and serving capabilities
- **📊 Performance Analytics**: Real-time performance monitoring and automated reporting
- **🛡️ Data Quality Validation**: Automated input validation and data quality assessment
- **📈 Model Interpretability**: Feature importance analysis and explainability support

## Project Structure

```
gymnasium/
├── src/                        # Core framework source code
│   ├── core/                   # Core framework components
│   │   ├── pipeline.py         # Main training pipeline
│   │   ├── base_model.py       # Base model interface
│   │   ├── trainer.py          # Training logic
│   │   ├── evaluator.py        # Evaluation logic
│   │   ├── inference.py        # 🔮 Advanced inference engine
│   │   ├── validation.py       # 🔍 Model validation framework
│   │   └── deployment.py       # 🚀 Deployment manager
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
│   ├── WANDB_SETUP_COMPLETE.md # WandB setup guide
│   └── INFERENCE_VALIDATION_FRAMEWORK.md # 📖 New framework docs
├── wandb_integration/          # Weights & Biases integration
│   ├── setup_wandb_server.py   # WandB server setup
│   ├── train_with_wandb.py     # Training with WandB
│   └── ...                     # Other WandB utilities
├── notebooks/                  # Jupyter notebooks for analysis
├── results/                    # Unified output directory (gitignored)
│   ├── <experiment_name>/      # Individual experiment folders
│   │   ├── models/            # Trained models and checkpoints
│   │   ├── logs/              # Training and validation logs
│   │   ├── visualizations/    # Generated plots and charts
│   │   └── metrics/           # Performance metrics and summaries
│   ├── inference/             # Inference results and monitoring
│   ├── validation/            # Model validation reports
│   └── deployment/            # Deployment artifacts and monitoring
├── inference_validation_demo.py # 🎯 Framework demonstration
└── test_framework.py           # 🧪 Framework tests
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run CPU allocation training:
```bash
python src/main.py --config_name=cpu_allocation
```

3. View results in TensorBoard:
```bash
tensorboard --logdir results/
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

### Basic Training Pipeline

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

### 🔮 Advanced Inference and Validation

```python
from src.core.inference import InferenceEngine
from src.core.validation import ModelValidator
from src.core.deployment import ModelDeploymentManager

# Advanced inference with monitoring
inference_engine = InferenceEngine(model, config)
inference_engine.set_reference_statistics(train_loader)

# Run inference with validation
results = inference_engine.run_inference(
    input_data,
    return_confidence=True,
    validate_inputs=True
)

# Comprehensive model validation
validator = ModelValidator(model, config)
validation_results = validator.comprehensive_validation(
    train_loader, val_loader, test_loader
)

# Production deployment
deployment_manager = ModelDeploymentManager(model, config)
deployment_results = deployment_manager.prepare_for_deployment(
    train_loader, val_loader, test_loader
)

# Start serving
deployment_manager.serve_model(enable_monitoring=True)
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

## 🚀 New: Inference and Validation Framework

The framework now includes comprehensive inference and validation capabilities for production deployment:

### Quick Test
```bash
# Test the framework
python test_framework.py

# Run full demonstration
python inference_validation_demo.py
```

### Key Features
- **🔮 InferenceEngine**: Advanced inference with monitoring, confidence scoring, and drift detection
- **🔍 ModelValidator**: Comprehensive validation pipelines with robustness testing
- **🚀 DeploymentManager**: Production-ready deployment with health monitoring
- **📊 Performance Analytics**: Real-time monitoring and automated reporting
- **🛡️ Data Validation**: Input quality assessment and validation
- **📈 Interpretability**: Feature importance and explainability support

### Documentation
- **📖 Complete Guide**: `docs/INFERENCE_VALIDATION_FRAMEWORK.md`
- **🎯 Demo Script**: `inference_validation_demo.py`
- **🧪 Tests**: `test_framework.py`

## Configuration

Models and training parameters are configured via YAML files in the `configs/` directory. See `configs/cpu_allocation.yaml` for an example.

### New Configuration Options
```yaml
# Inference configuration
inference:
  batch_size: 32
  confidence_threshold: 0.8
  drift_threshold: 0.1

# Validation configuration  
validation:
  degradation_threshold: 0.05
  consistency_score: 0.8

# Deployment configuration
deployment:
  enable_monitoring: true
  monitoring_interval: 3600
```

## License

MIT License

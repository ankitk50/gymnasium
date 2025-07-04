# ML/DL Training Pipeline Framework

## Quick Start

This framework provides a complete solution for training machine learning and deep learning models, with specialized support for CPU allocation tasks.

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Train a model with default configuration:**
```bash
python main.py
```

2. **Train with custom configuration:**
```bash
python main.py --config configs/cpu_allocation_lstm.yaml
```

3. **Run examples:**
```bash
python examples.py
```

### Available Models

- **MLP**: `cpu_allocation_mlp` - Basic feedforward neural network
- **LSTM**: `cpu_allocation_lstm` - Recurrent network for temporal patterns
- **Transformer**: `cpu_allocation_transformer` - Attention-based model

### Configuration Files

- `configs/cpu_allocation.yaml` - Basic MLP configuration
- `configs/cpu_allocation_lstm.yaml` - LSTM configuration
- `configs/cpu_allocation_transformer.yaml` - Transformer configuration

### Key Features

✅ **Generic Framework**: Supports both traditional ML and deep learning models  
✅ **CPU Allocation Models**: Specialized models for CPU resource allocation  
✅ **Comprehensive Visualization**: Training metrics, evaluation plots, and interactive dashboards  
✅ **Multiple Evaluation Methods**: Standard metrics, cross-validation, hyperparameter optimization  
✅ **Configuration Management**: YAML-based configuration with easy overrides  
✅ **Experiment Tracking**: Automatic logging and result saving  

### Project Structure

```
src/
├── core/           # Core framework components
├── models/         # Model implementations  
├── data/           # Data handling utilities
├── visualization/ # Visualization components
└── utils/          # Utility functions

configs/           # Configuration files
experiments/       # Experiment outputs (auto-generated)
```

### Results

After training, check the `experiments/` directory for:
- Training curves and metrics
- Model checkpoints
- Evaluation results and visualizations
- Interactive HTML dashboards

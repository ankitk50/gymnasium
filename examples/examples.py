"""
Example usage of the ML/DL training pipeline framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.pipeline import TrainingPipeline
from src.utils.config import create_default_config
from src.data.data_loader import create_synthetic_cpu_allocation_data
import pandas as pd


def basic_training_example():
    """Basic example of training a CPU allocation model."""
    print("=" * 60)
    print("BASIC TRAINING EXAMPLE")
    print("=" * 60)
    
    # Create default configuration
    config = create_default_config()
    config.set('experiment_name', 'basic_example')
    config.set('training.epochs', 50)  # Shorter training for example
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        config_dict=config.to_dict(),
        experiment_name='basic_example'
    )
    
    # Train model
    print("Training model...")
    trained_model = pipeline.train()
    
    # Evaluate model
    print("Evaluating model...")
    results = pipeline.evaluate(trained_model)
    
    print(f"Training completed! Results saved to: {pipeline.output_dir}")
    print(f"Final MSE: {results['metrics']['mse']:.4f}")
    print(f"Final R²: {results['metrics']['r2_score']:.4f}")


def lstm_training_example():
    """Example of training an LSTM model for temporal CPU allocation."""
    print("=" * 60)
    print("LSTM TRAINING EXAMPLE")
    print("=" * 60)
    
    # Configuration for LSTM model
    config = create_default_config()
    config.set('experiment_name', 'lstm_example')
    config.set('model.name', 'cpu_allocation_lstm')
    config.set('data.sequence_length', 10)  # Enable temporal sequences
    config.set('training.epochs', 30)
    
    # Update model configuration for LSTM
    config.set('model.hidden_dim', 64)
    config.set('model.num_layers', 2)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        config_dict=config.to_dict(),
        experiment_name='lstm_example'
    )
    
    # Train model
    print("Training LSTM model...")
    trained_model = pipeline.train()
    
    # Evaluate model
    print("Evaluating LSTM model...")
    results = pipeline.evaluate(trained_model)
    
    print(f"LSTM training completed! Results saved to: {pipeline.output_dir}")
    print(f"Final MSE: {results['metrics']['mse']:.4f}")
    print(f"Final R²: {results['metrics']['r2_score']:.4f}")


def cross_validation_example():
    """Example of cross-validation."""
    print("=" * 60)
    print("CROSS-VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config()
    config.set('experiment_name', 'cv_example')
    config.set('training.epochs', 20)  # Shorter epochs for CV
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        config_dict=config.to_dict(),
        experiment_name='cv_example'
    )
    
    # Run cross-validation
    print("Running 5-fold cross-validation...")
    cv_results = pipeline.cross_validate(k_folds=5)
    
    print("Cross-validation completed!")
    print("CV Results:")
    for metric, value in cv_results['cv_metrics'].items():
        print(f"  {metric}: {value:.4f}")


def custom_data_example():
    """Example using custom synthetic data."""
    print("=" * 60)
    print("CUSTOM DATA EXAMPLE")
    print("=" * 60)
    
    # Create custom synthetic data
    print("Generating custom synthetic data...")
    df = create_synthetic_cpu_allocation_data(
        n_samples=5000,
        n_features=8,
        task_type='regression',
        noise_level=0.05
    )
    
    # Save data to file
    data_path = Path('data/custom_cpu_allocation.csv')
    data_path.parent.mkdir(exist_ok=True)
    df.to_csv(data_path, index=False)
    
    # Create configuration to use custom data
    config = create_default_config()
    config.set('experiment_name', 'custom_data_example')
    config.set('data.source', str(data_path))
    config.set('data.n_features', 8)
    config.set('model.input_dim', 8)
    config.set('training.epochs', 40)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        config_dict=config.to_dict(),
        experiment_name='custom_data_example'
    )
    
    # Train model
    print("Training with custom data...")
    trained_model = pipeline.train()
    
    # Evaluate model
    print("Evaluating with custom data...")
    results = pipeline.evaluate(trained_model)
    
    print(f"Custom data training completed! Results saved to: {pipeline.output_dir}")
    print(f"Final MSE: {results['metrics']['mse']:.4f}")
    print(f"Final R²: {results['metrics']['r2_score']:.4f}")


def hyperparameter_optimization_example():
    """Example of hyperparameter optimization."""
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config()
    config.set('experiment_name', 'hp_optimization_example')
    config.set('training.epochs', 30)  # Shorter epochs for HP search
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        config_dict=config.to_dict(),
        experiment_name='hp_optimization_example'
    )
    
    # Define search space
    search_space = {
        'model.learning_rate': {'type': 'float', 'range': [1e-4, 1e-2]},
        'model.hidden_dims.0': {'type': 'int', 'range': [64, 256]},
        'model.hidden_dims.1': {'type': 'int', 'range': [32, 128]},
        'model.dropout_rate': {'type': 'float', 'range': [0.1, 0.5]},
    }
    
    print("Running hyperparameter optimization...")
    hp_results = pipeline.hyperparameter_search(search_space, n_trials=10)
    
    print("Hyperparameter optimization completed!")
    print("Best parameters:")
    for param, value in hp_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best validation loss: {hp_results['best_value']:.4f}")


def inference_example():
    """Example of model inference."""
    print("=" * 60)
    print("INFERENCE EXAMPLE")
    print("=" * 60)
    
    # First train a model quickly
    config = create_default_config()
    config.set('experiment_name', 'inference_example_training')
    config.set('training.epochs', 20)
    
    pipeline = TrainingPipeline(
        config_dict=config.to_dict(),
        experiment_name='inference_example_training'
    )
    
    print("Training model for inference...")
    trained_model = pipeline.train()
    
    # Now use for inference
    print("Running inference...")
    pipeline.setup_data()
    inference_results = pipeline.inference(pipeline.test_loader, trained_model)
    
    print(f"Inference completed on {inference_results['num_samples']} samples")
    print(f"Sample predictions: {inference_results['predictions'][:5]}")


def main():
    """Run all examples."""
    try:
        # Run basic example
        basic_training_example()
        
        # Run LSTM example
        lstm_training_example()
        
        # Run cross-validation example
        cross_validation_example()
        
        # Run custom data example
        custom_data_example()
        
        # Run inference example
        inference_example()
        
        # Hyperparameter optimization example (commented out as it takes longer)
        # hyperparameter_optimization_example()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nCheck the 'experiments/' directory for results and visualizations.")
        print("Open the generated HTML files in your browser to view interactive plots.")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

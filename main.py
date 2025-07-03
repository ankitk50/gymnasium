"""
Main entry point for the ML/DL training pipeline framework.
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.pipeline import TrainingPipeline
from src.utils.config import Config, create_default_config
from src.utils.logging import setup_logging


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='ML/DL Training Pipeline Framework')
    
    parser.add_argument('--config', '-c', type=str, default='configs/cpu_allocation.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', '-e', type=str, default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation (requires existing model)')
    parser.add_argument('--cross-validate', action='store_true',
                       help='Run cross-validation')
    parser.add_argument('--hyperparameter-search', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--model-checkpoint', type=str, default=None,
                       help='Path to model checkpoint to load')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level=log_level)
    
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
            experiment_name=config.get('experiment_name')
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
            
            print("Pipeline completed successfully!")
            print(f"Results saved to: {pipeline.output_dir}")
            
            # Print key metrics
            metrics = results.get('metrics', {})
            if 'mse' in metrics:
                print(f"Final MSE: {metrics['mse']:.4f}")
            if 'mae' in metrics:
                print(f"Final MAE: {metrics['mae']:.4f}")
            if 'r2_score' in metrics:
                print(f"Final R²: {metrics['r2_score']:.4f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

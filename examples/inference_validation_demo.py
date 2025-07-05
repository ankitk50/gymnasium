"""
Example demonstrating the comprehensive inference and validation framework.
Shows how to use the new inference, validation, and deployment capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.base_model import BaseModel
from src.core.inference import InferenceEngine
from src.core.validation import ModelValidator
from src.core.deployment import ModelDeploymentManager
from src.data.data_loader import create_data_loaders
from src.utils.config import create_default_config


class ExampleModel(BaseModel):
    """Example model for demonstration."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Simple MLP architecture
        input_size = config.get('input_size', 10)
        hidden_size = config.get('hidden_size', 64)
        output_size = config.get('output_size', 1)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
    
    def forward(self, x):
        return self.layers(x)
    
    def get_loss_function(self):
        return nn.MSELoss()
    
    def get_optimizer(self, parameters):
        lr = self.config.get('learning_rate', 0.001)
        return torch.optim.Adam(parameters, lr=lr)


def create_sample_data(num_samples=1000, input_size=10, noise_level=0.1):
    """Create sample data for demonstration."""
    # Generate synthetic data
    X = np.random.randn(num_samples, input_size)
    
    # Create a non-linear relationship
    y = (np.sum(X[:, :3], axis=1) + 
         0.5 * np.sum(X[:, 3:6] ** 2, axis=1) + 
         noise_level * np.random.randn(num_samples))
    
    return X.astype(np.float32), y.astype(np.float32)


def demonstrate_inference_framework():
    """Demonstrate the inference framework capabilities."""
    
    print("🚀 INFERENCE AND VALIDATION FRAMEWORK DEMO")
    print("=" * 60)
    
    # Configuration
    config = {
        'input_size': 10,
        'hidden_size': 64,
        'output_size': 1,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'task_type': 'regression',
        'device': 'cpu',
        'input_shape': [10],
        'output_shape': [1],
        'validation': {
            'confidence_threshold': 0.8,
            'drift_threshold': 0.1,
            'degradation_threshold': 0.05
        },
        'validation_criteria': {
            'minimum_r2_score': 0.6,
            'maximum_mse': 1.0,
            'maximum_inference_time': 1.0,
            'minimum_consistency_score': 0.8
        }
    }
    
    print("📋 Configuration loaded")
    print(f"   Input size: {config['input_size']}")
    print(f"   Task type: {config['task_type']}")
    print(f"   Device: {config['device']}")
    
    # Create model
    print("\n🔧 Creating and training model...")
    model = ExampleModel(config)
    print(f"   Model parameters: {model.count_parameters():,}")
    
    # Create sample data
    print("\n📊 Generating sample data...")
    X_train, y_train = create_sample_data(800, config['input_size'])
    X_val, y_val = create_sample_data(100, config['input_size'])
    X_test, y_test = create_sample_data(100, config['input_size'])
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Quick training (for demonstration)
    print("\n🏋️ Quick model training...")
    model.train()
    optimizer = model.get_optimizer(model.parameters())
    criterion = model.get_loss_function()
    
    for epoch in range(20):  # Quick training
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(model.device), batch_y.to(model.device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"   Epoch {epoch + 1}/20 - Loss: {avg_loss:.4f}")
    
    print("✅ Model training completed")
    
    # 1. INFERENCE ENGINE DEMONSTRATION
    print("\n" + "=" * 60)
    print("🔮 INFERENCE ENGINE DEMONSTRATION")
    print("=" * 60)
    
    inference_engine = InferenceEngine(model, config, 'demo_output/inference')
    
    # Set reference statistics
    print("\n📈 Setting reference statistics for drift detection...")
    inference_engine.set_reference_statistics(train_loader)
    print("✅ Reference statistics set")
    
    # Single inference
    print("\n🎯 Running single inference...")
    sample_input = torch.randn(5, config['input_size'])
    
    inference_result = inference_engine.run_inference(
        sample_input,
        return_confidence=True,
        return_features=True,
        validate_inputs=True
    )
    
    print(f"   Predictions: {len(inference_result['predictions'])} samples")
    print(f"   Inference time: {inference_result['inference_time']:.4f}s")
    print(f"   Throughput: {inference_result['throughput']:.1f} samples/sec")
    print(f"   Input validation: {'✅ Passed' if inference_result['input_validation']['valid'] else '❌ Failed'}")
    print(f"   Drift detected: {'⚠️ Yes' if inference_result['drift_analysis']['drift_detected'] else '✅ No'}")
    
    if 'confidence_scores' in inference_result:
        avg_confidence = inference_result['avg_confidence']
        print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Batch inference
    print("\n📦 Running batch inference...")
    batch_data = [torch.randn(3, config['input_size']) for _ in range(5)]
    
    batch_results, aggregated = inference_engine.batch_inference(
        batch_data, parallel=True, max_workers=2
    )
    
    print(f"   Processed {len(batch_results)} batches")
    print(f"   Total samples: {aggregated['total_samples']}")
    print(f"   Total time: {aggregated['total_time']:.4f}s")
    print(f"   Average throughput: {aggregated['average_throughput']:.1f} samples/sec")
    
    # Performance summary
    performance_summary = inference_engine.get_performance_summary()
    print(f"\n📊 Performance Summary:")
    print(f"   Total inferences: {performance_summary['total_inferences']}")
    print(f"   Average inference time: {performance_summary['average_inference_time']:.4f}s")
    print(f"   Average throughput: {performance_summary['average_throughput']:.1f} samples/sec")
    
    # 2. MODEL VALIDATION DEMONSTRATION
    print("\n" + "=" * 60)
    print("🔍 MODEL VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    validator = ModelValidator(model, config, 'demo_output/validation')
    
    # Data quality validation
    print("\n🔎 Validating data quality...")
    data_quality = validator.validate_data_quality(test_loader)
    
    print(f"   Quality score: {data_quality['quality_score']:.2f}/1.0")
    print(f"   Issues found: {len(data_quality['issues'])}")
    print(f"   Warnings: {len(data_quality['warnings'])}")
    
    if data_quality['issues']:
        for issue in data_quality['issues']:
            print(f"   ❌ {issue}")
    
    if data_quality['warnings']:
        for warning in data_quality['warnings']:
            print(f"   ⚠️ {warning}")
    
    # Model interpretability
    print("\n🧠 Validating model interpretability...")
    interpretability = validator.validate_model_interpretability(test_loader)
    
    print(f"   Feature importance available: {'✅' if interpretability['feature_importance_available'] else '❌'}")
    print(f"   Prediction explainability: {interpretability['prediction_explainability']}")
    print(f"   Model complexity: {interpretability['model_complexity']['complexity_level']}")
    
    # Comprehensive validation
    print("\n🎯 Running comprehensive validation...")
    validation_results = validator.comprehensive_validation(
        train_loader, val_loader, test_loader
    )
    
    summary = validation_results['validation_summary']
    print(f"   Overall status: {summary['overall_status']}")
    print(f"   Quality score: {summary['quality_score']:.2f}/1.0")
    print(f"   Tests passed: {len(summary['passed_tests'])}")
    print(f"   Tests failed: {len(summary['failed_tests'])}")
    print(f"   Warnings: {len(summary['warnings'])}")
    
    if summary['passed_tests']:
        print("   ✅ Passed tests:")
        for test in summary['passed_tests']:
            print(f"      - {test}")
    
    if summary['failed_tests']:
        print("   ❌ Failed tests:")
        for test in summary['failed_tests']:
            print(f"      - {test}")
    
    if summary['recommendations']:
        print("   💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"      - {rec}")
    
    # 3. DEPLOYMENT MANAGER DEMONSTRATION
    print("\n" + "=" * 60)
    print("🚀 DEPLOYMENT MANAGER DEMONSTRATION")
    print("=" * 60)
    
    deployment_manager = ModelDeploymentManager(model, config, 'demo_output/deployment')
    
    # Prepare for deployment
    print("\n📋 Preparing model for deployment...")
    preparation_results = deployment_manager.prepare_for_deployment(
        train_loader, val_loader, test_loader
    )
    
    print(f"   Preparation status: {preparation_results['status']}")
    print(f"   Steps completed: {len(preparation_results['steps_completed'])}")
    print(f"   Steps failed: {len(preparation_results['steps_failed'])}")
    print(f"   Deployment ready: {'✅' if preparation_results['deployment_ready'] else '❌'}")
    
    if preparation_results['deployment_ready']:
        # Start serving
        print("\n🌐 Starting model serving...")
        deployment_manager.serve_model(enable_monitoring=True)
        
        # Make some predictions
        print("\n🎯 Making test predictions...")
        
        for i in range(5):
            test_input = np.random.randn(1, config['input_size'])
            prediction_result = deployment_manager.predict(
                test_input,
                return_confidence=True,
                validate_input=True
            )
            
            if 'error' not in prediction_result:
                pred_value = prediction_result['predictions'][0]
                confidence = prediction_result.get('avg_confidence', 'N/A')
                response_time = prediction_result['response_time']
                print(f"   Request {i+1}: pred={pred_value:.4f}, conf={confidence}, time={response_time:.4f}s")
            else:
                print(f"   Request {i+1}: ❌ {prediction_result['error']}")
        
        # Get deployment status
        deployment_status = deployment_manager.get_deployment_status()
        serving_stats = deployment_status['serving_stats']
        
        print(f"\n📊 Deployment Statistics:")
        print(f"   Total requests: {serving_stats['total_requests']}")
        print(f"   Successful predictions: {serving_stats['successful_predictions']}")
        print(f"   Failed predictions: {serving_stats['failed_predictions']}")
        print(f"   Average response time: {serving_stats['average_response_time']:.4f}s")
        print(f"   Uptime: {deployment_status['uptime_formatted']}")
        
        # Validate deployment
        print("\n🔍 Validating deployment...")
        deployment_validation = deployment_manager.validate_deployment(test_loader)
        
        print(f"   Validation status: {deployment_validation['status']}")
        if 'performance_degradation_detected' in deployment_validation:
            degradation = deployment_validation['performance_degradation_detected']
            print(f"   Performance degradation: {'⚠️ Detected' if degradation else '✅ None'}")
        
        # Generate reports
        print("\n📝 Generating deployment report...")
        report_path = deployment_manager.generate_deployment_report()
        print(f"   Report saved to: {report_path}")
        
        # Export for production
        print("\n📦 Exporting for production...")
        exported_files = deployment_manager.export_model_for_production(
            export_format='torch',
            include_pipeline=True
        )
        
        print("   Exported files:")
        for file_type, file_path in exported_files.items():
            print(f"      {file_type}: {file_path}")
        
        # Shutdown
        print("\n🛑 Shutting down deployment...")
        deployment_manager.shutdown()
    
    # 4. ADVANCED FEATURES DEMONSTRATION
    print("\n" + "=" * 60)
    print("⚡ ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Model export in different formats
    print("\n💾 Testing model export capabilities...")
    
    # Export as PyTorch
    torch_export = inference_engine.export_inference_model(
        'demo_output/exports/torch', 'torch', include_metadata=True
    )
    print(f"   PyTorch export: {torch_export}")
    
    # Export as TorchScript
    try:
        torchscript_export = inference_engine.export_inference_model(
            'demo_output/exports/torchscript', 'torchscript', include_metadata=True
        )
        print(f"   TorchScript export: {torchscript_export}")
    except Exception as e:
        print(f"   TorchScript export failed: {e}")
    
    # Performance benchmarking
    print("\n⚡ Performance benchmarking...")
    benchmark_results = inference_engine.evaluator.benchmark_inference(test_loader, num_runs=10)
    
    print(f"   Mean inference time: {benchmark_results['mean_inference_time']:.4f}s")
    print(f"   Std inference time: {benchmark_results['std_inference_time']:.4f}s")
    print(f"   Min inference time: {benchmark_results['min_inference_time']:.4f}s")
    print(f"   Max inference time: {benchmark_results['max_inference_time']:.4f}s")
    
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Output files saved to: demo_output/")
    print("📋 Check the generated reports and visualizations")
    print("🚀 Framework ready for production use!")


if __name__ == "__main__":
    demonstrate_inference_framework()

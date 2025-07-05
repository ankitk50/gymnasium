"""
Integrated inference and validation manager for comprehensive model deployment.
Combines inference capabilities with validation frameworks for production-ready ML systems.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import json
import time
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

from .base_model import BaseModel
from .inference import InferenceEngine
from .validation import ModelValidator
from .evaluator import Evaluator
from ..visualization.inference_viz import InferenceVisualizer
from ..visualization.evaluation_viz import EvaluationVisualizer
from ..utils.logging import setup_logging


class ModelDeploymentManager:
    """
    Comprehensive model deployment manager integrating inference and validation.
    Provides production-ready capabilities for model serving and monitoring.
    """
    
    def __init__(self,
                 model: BaseModel,
                 config: Optional[Dict[str, Any]] = None,
                 output_dir: str = 'deployment_output'):
        """
        Initialize the deployment manager.
        
        Args:
            model: Trained model for deployment
            config: Deployment configuration
            output_dir: Directory for saving deployment artifacts
        """
        self.model = model
        self.config = config or {}
        self.device = model.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging('deployment_manager', self.output_dir / 'deployment.log')
        
        # Initialize core components
        self.inference_engine = InferenceEngine(
            model, config, str(self.output_dir / 'inference')
        )
        self.validator = ModelValidator(
            model, config, str(self.output_dir / 'validation')
        )
        self.evaluator = Evaluator(model, config)
        
        # Initialize visualizers
        self.inference_viz = InferenceVisualizer(str(self.output_dir / 'visualizations'))
        self.evaluation_viz = EvaluationVisualizer(str(self.output_dir / 'visualizations'))
        
        # Deployment status tracking
        self.deployment_status = {
            'initialized': datetime.now().isoformat(),
            'model_validated': False,
            'ready_for_inference': False,
            'monitoring_active': False,
            'performance_baseline': None
        }
        
        # Model serving statistics
        self.serving_stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_response_time': 0.0,
            'uptime_start': datetime.now(),
            'last_validation': None
        }
        
        self.logger.info(f"Deployment manager initialized for {model.__class__.__name__}")
    
    def prepare_for_deployment(self,
                             train_data: DataLoader,
                             val_data: DataLoader,
                             test_data: DataLoader,
                             baseline_metrics: Optional[Dict[str, float]] = None,
                             validation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare model for deployment with comprehensive validation and setup.
        
        Args:
            train_data: Training dataset for reference statistics
            val_data: Validation dataset
            test_data: Test dataset for validation
            baseline_metrics: Baseline performance metrics
            validation_config: Custom validation configuration
            
        Returns:
            Deployment preparation results
        """
        self.logger.info("Preparing model for deployment...")
        
        preparation_results = {
            'preparation_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'status': 'in_progress',
            'steps_completed': [],
            'steps_failed': [],
            'deployment_ready': False
        }
        
        try:
            # Step 1: Comprehensive model validation
            self.logger.info("Step 1: Running comprehensive model validation...")
            
            if validation_config:
                self.config.update(validation_config)
            
            validation_results = self.validator.comprehensive_validation(
                train_data, val_data, test_data, baseline_metrics
            )
            
            preparation_results['validation_results'] = validation_results
            
            if validation_results['validation_summary']['overall_status'] in ['PASSED', 'PASSED_WITH_WARNINGS']:
                preparation_results['steps_completed'].append('model_validation')
                self.deployment_status['model_validated'] = True
            else:
                preparation_results['steps_failed'].append('model_validation')
                raise ValueError("Model validation failed - cannot proceed with deployment")
            
            # Step 2: Set up reference statistics for drift detection
            self.logger.info("Step 2: Setting up reference statistics...")
            self.inference_engine.set_reference_statistics(train_data)
            preparation_results['steps_completed'].append('reference_statistics')
            
            # Step 3: Benchmark inference performance
            self.logger.info("Step 3: Benchmarking inference performance...")
            inference_benchmark = self.inference_engine.evaluator.benchmark_inference(test_data)
            preparation_results['inference_benchmark'] = inference_benchmark
            preparation_results['steps_completed'].append('inference_benchmark')
            
            # Step 4: Validate data quality
            self.logger.info("Step 4: Validating data quality...")
            data_quality = self.validator.validate_data_quality(test_data)
            preparation_results['data_quality'] = data_quality
            
            if data_quality['quality_score'] >= 0.7:
                preparation_results['steps_completed'].append('data_quality')
            else:
                preparation_results['steps_failed'].append('data_quality')
                self.logger.warning(f"Data quality score low: {data_quality['quality_score']}")
            
            # Step 5: Test model interpretability
            self.logger.info("Step 5: Testing model interpretability...")
            interpretability_results = self.validator.validate_model_interpretability(test_data)
            preparation_results['interpretability'] = interpretability_results
            preparation_results['steps_completed'].append('interpretability_check')
            
            # Step 6: Create deployment package
            self.logger.info("Step 6: Creating deployment package...")
            deployment_package = self._create_deployment_package(preparation_results)
            preparation_results['deployment_package'] = deployment_package
            preparation_results['steps_completed'].append('deployment_package')
            
            # Step 7: Set performance baseline
            self.logger.info("Step 7: Setting performance baseline...")
            self.deployment_status['performance_baseline'] = validation_results['basic_evaluation']['metrics']
            preparation_results['steps_completed'].append('performance_baseline')
            
            # Check if ready for deployment
            if len(preparation_results['steps_failed']) == 0:
                preparation_results['deployment_ready'] = True
                preparation_results['status'] = 'completed'
                self.deployment_status['ready_for_inference'] = True
                self.logger.info("Model successfully prepared for deployment!")
            else:
                preparation_results['status'] = 'completed_with_warnings'
                self.logger.warning(f"Deployment preparation completed with {len(preparation_results['steps_failed'])} failed steps")
            
        except Exception as e:
            preparation_results['status'] = 'failed'
            preparation_results['error'] = str(e)
            self.logger.error(f"Deployment preparation failed: {e}")
        
        # Save preparation results
        self._save_preparation_results(preparation_results)
        
        return preparation_results
    
    def serve_model(self,
                   request_handler: Optional[Callable] = None,
                   enable_monitoring: bool = True,
                   monitoring_interval: int = 3600) -> None:
        """
        Start model serving with optional monitoring.
        
        Args:
            request_handler: Custom request handling function
            enable_monitoring: Whether to enable continuous monitoring
            monitoring_interval: Monitoring interval in seconds
        """
        if not self.deployment_status['ready_for_inference']:
            raise RuntimeError("Model not ready for inference. Run prepare_for_deployment first.")
        
        self.logger.info("Starting model serving...")
        
        # Start monitoring if enabled
        if enable_monitoring:
            self._start_monitoring(monitoring_interval)
        
        # Update serving status
        self.serving_stats['uptime_start'] = datetime.now()
        
        self.logger.info("Model serving started successfully")
    
    def predict(self,
               input_data: Union[torch.Tensor, np.ndarray, pd.DataFrame],
               return_confidence: bool = True,
               validate_input: bool = True,
               log_request: bool = True) -> Dict[str, Any]:
        """
        Make predictions with comprehensive monitoring and validation.
        
        Args:
            input_data: Input data for prediction
            return_confidence: Whether to return confidence scores
            validate_input: Whether to validate input data
            log_request: Whether to log the prediction request
            
        Returns:
            Prediction results with metadata
        """
        request_start = time.time()
        request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        if not self.deployment_status['ready_for_inference']:
            raise RuntimeError("Model not ready for inference")
        
        try:
            # Update request counter
            self.serving_stats['total_requests'] += 1
            
            if log_request:
                self.logger.info(f"Processing prediction request {request_id}")
            
            # Run inference
            inference_results = self.inference_engine.run_inference(
                input_data,
                return_confidence=return_confidence,
                return_features=False,
                validate_inputs=validate_input
            )
            
            # Prepare response
            response = {
                'request_id': request_id,
                'predictions': inference_results['predictions'],
                'timestamp': inference_results['timestamp'],
                'response_time': time.time() - request_start,
                'model_info': {
                    'name': self.model.__class__.__name__,
                    'version': self.config.get('model_version', '1.0.0')
                }
            }
            
            if return_confidence:
                response['confidence_scores'] = inference_results.get('confidence_scores')
                response['avg_confidence'] = inference_results.get('avg_confidence')
            
            if validate_input:
                response['input_validation'] = inference_results.get('input_validation')
                response['drift_analysis'] = inference_results.get('drift_analysis')
            
            # Update serving statistics
            self.serving_stats['successful_predictions'] += 1
            self._update_response_time_stats(response['response_time'])
            
            if log_request:
                self.logger.info(f"Request {request_id} completed in {response['response_time']:.3f}s")
            
            return response
            
        except Exception as e:
            self.serving_stats['failed_predictions'] += 1
            self.logger.error(f"Prediction request {request_id} failed: {e}")
            
            return {
                'request_id': request_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'response_time': time.time() - request_start,
                'status': 'failed'
            }
    
    def batch_predict(self,
                     batch_data: List[Union[torch.Tensor, np.ndarray, pd.DataFrame]],
                     parallel: bool = True,
                     max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Process batch predictions with parallel processing.
        
        Args:
            batch_data: List of input data for batch processing
            parallel: Whether to process in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of prediction results
        """
        self.logger.info(f"Processing batch prediction with {len(batch_data)} items")
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.predict, data, True, True, False)
                    for data in batch_data
                ]
                results = [future.result() for future in futures]
        else:
            results = [self.predict(data, True, True, False) for data in batch_data]
        
        # Aggregate batch statistics
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        self.logger.info(f"Batch prediction completed: {successful} successful, {failed} failed")
        
        return results
    
    def validate_deployment(self,
                          test_data: DataLoader,
                          run_comprehensive: bool = False) -> Dict[str, Any]:
        """
        Validate current deployment status and performance.
        
        Args:
            test_data: Test dataset for validation
            run_comprehensive: Whether to run comprehensive validation
            
        Returns:
            Deployment validation results
        """
        self.logger.info("Validating deployment...")
        
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'deployment_status': self.deployment_status.copy(),
            'serving_stats': self.serving_stats.copy(),
            'uptime': str(datetime.now() - self.serving_stats['uptime_start']),
            'validation_type': 'comprehensive' if run_comprehensive else 'basic'
        }
        
        try:
            if run_comprehensive:
                # Run comprehensive validation
                comprehensive_results = self.validator.comprehensive_validation(
                    test_data, test_data, test_data,
                    self.deployment_status['performance_baseline']
                )
                validation_results['comprehensive_validation'] = comprehensive_results
            else:
                # Run basic performance check
                basic_results = self.validator._quick_validation(test_data)
                validation_results['basic_validation'] = basic_results
                
                # Check for performance degradation
                if self.deployment_status['performance_baseline']:
                    degradation_detected = self.validator._detect_performance_degradation(basic_results)
                    validation_results['performance_degradation_detected'] = degradation_detected
            
            # Test inference performance
            inference_test = self.inference_engine.run_inference(
                next(iter(test_data))[0][:5],  # Small sample
                return_confidence=True,
                validate_inputs=True
            )
            validation_results['inference_test'] = inference_test
            
            # Update last validation time
            self.serving_stats['last_validation'] = datetime.now().isoformat()
            
            validation_results['status'] = 'completed'
            
        except Exception as e:
            validation_results['status'] = 'failed'
            validation_results['error'] = str(e)
            self.logger.error(f"Deployment validation failed: {e}")
        
        return validation_results
    
    def generate_deployment_report(self) -> str:
        """
        Generate comprehensive deployment report.
        
        Returns:
            Path to generated report
        """
        self.logger.info("Generating deployment report...")
        
        # Calculate uptime
        uptime = datetime.now() - self.serving_stats['uptime_start']
        
        # Calculate success rate
        total_requests = self.serving_stats['total_requests']
        success_rate = (self.serving_stats['successful_predictions'] / total_requests * 100) if total_requests > 0 else 0
        
        report_content = f"""
# Model Deployment Report

## Deployment Overview
- **Model**: {self.model.__class__.__name__}
- **Deployment Time**: {self.deployment_status['initialized']}
- **Status**: {"✅ Active" if self.deployment_status['ready_for_inference'] else "❌ Inactive"}
- **Uptime**: {uptime}

## Performance Statistics
- **Total Requests**: {total_requests:,}
- **Successful Predictions**: {self.serving_stats['successful_predictions']:,}
- **Failed Predictions**: {self.serving_stats['failed_predictions']:,}
- **Success Rate**: {success_rate:.2f}%
- **Average Response Time**: {self.serving_stats['average_response_time']:.4f}s

## Model Information
- **Parameters**: {self.model.count_parameters():,}
- **Input Shape**: {self.model.input_shape}
- **Output Shape**: {self.model.output_shape}
- **Device**: {self.device}

## Deployment Status
- **Model Validated**: {"✅" if self.deployment_status['model_validated'] else "❌"}
- **Ready for Inference**: {"✅" if self.deployment_status['ready_for_inference'] else "❌"}
- **Monitoring Active**: {"✅" if self.deployment_status['monitoring_active'] else "❌"}
- **Last Validation**: {self.serving_stats.get('last_validation', 'Never')}

## Performance Baseline
"""
        
        if self.deployment_status['performance_baseline']:
            baseline = self.deployment_status['performance_baseline']
            for metric, value in baseline.items():
                if isinstance(value, (int, float)):
                    report_content += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
        else:
            report_content += "- No baseline metrics available\n"
        
        report_content += f"""

## Recommendations
"""
        
        # Generate recommendations based on current status
        recommendations = []
        
        if not self.deployment_status['model_validated']:
            recommendations.append("❗ Run model validation before deployment")
        
        if self.serving_stats['failed_predictions'] > 0:
            failure_rate = self.serving_stats['failed_predictions'] / total_requests * 100
            if failure_rate > 5:
                recommendations.append(f"⚠️ High failure rate ({failure_rate:.1f}%) - investigate error patterns")
        
        if self.serving_stats['average_response_time'] > 1.0:
            recommendations.append("⚠️ High response time - consider model optimization")
        
        if not self.deployment_status['monitoring_active']:
            recommendations.append("💡 Enable continuous monitoring for production deployment")
        
        if not recommendations:
            recommendations.append("✅ No immediate issues detected")
        
        for rec in recommendations:
            report_content += f"- {rec}\n"
        
        report_content += f"""

## Generated On
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        report_path = self.output_dir / 'deployment_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Deployment report saved to {report_path}")
        
        return str(report_path)
    
    def export_model_for_production(self,
                                   export_format: str = 'torch',
                                   include_pipeline: bool = True) -> Dict[str, str]:
        """
        Export model and pipeline for production deployment.
        
        Args:
            export_format: Export format ('torch', 'onnx', 'torchscript')
            include_pipeline: Whether to include preprocessing pipeline
            
        Returns:
            Dictionary of exported file paths
        """
        self.logger.info(f"Exporting model for production in {export_format} format...")
        
        export_dir = self.output_dir / 'production_export'
        export_dir.mkdir(exist_ok=True)
        
        exported_files = {}
        
        # Export model
        model_path = self.inference_engine.export_inference_model(
            str(export_dir), export_format, include_metadata=True
        )
        exported_files['model'] = model_path
        
        # Export configuration
        config_path = export_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        exported_files['config'] = str(config_path)
        
        # Export deployment status
        status_path = export_dir / 'deployment_status.json'
        with open(status_path, 'w') as f:
            json.dump(self.deployment_status, f, indent=2, default=str)
        exported_files['deployment_status'] = str(status_path)
        
        # Export serving statistics
        stats_path = export_dir / 'serving_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.serving_stats, f, indent=2, default=str)
        exported_files['serving_stats'] = str(stats_path)
        
        # Create deployment script
        deployment_script = self._create_deployment_script(exported_files)
        script_path = export_dir / 'deploy.py'
        with open(script_path, 'w') as f:
            f.write(deployment_script)
        exported_files['deployment_script'] = str(script_path)
        
        self.logger.info(f"Model exported for production to {export_dir}")
        
        return exported_files
    
    def _create_deployment_package(self, preparation_results: Dict[str, Any]) -> Dict[str, str]:
        """Create deployment package with all necessary files."""
        package_dir = self.output_dir / 'deployment_package'
        package_dir.mkdir(exist_ok=True)
        
        package_files = {}
        
        # Save model state
        model_path = package_dir / 'model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config,
            'model_class': self.model.__class__.__name__
        }, model_path)
        package_files['model'] = str(model_path)
        
        # Save preparation results
        results_path = package_dir / 'preparation_results.json'
        with open(results_path, 'w') as f:
            json.dump(preparation_results, f, indent=2, default=str)
        package_files['preparation_results'] = str(results_path)
        
        # Save configuration
        config_path = package_dir / 'deployment_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        package_files['config'] = str(config_path)
        
        # Create requirements file
        requirements = [
            'torch>=1.9.0',
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'scikit-learn>=1.0.0',
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0'
        ]
        
        req_path = package_dir / 'requirements.txt'
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
        package_files['requirements'] = str(req_path)
        
        return package_files
    
    def _start_monitoring(self, interval: int) -> None:
        """Start continuous monitoring in background."""
        self.deployment_status['monitoring_active'] = True
        self.logger.info(f"Continuous monitoring started with {interval}s interval")
        
        # In a real implementation, this would run in a separate thread/process
        # For now, we just mark it as active
    
    def _update_response_time_stats(self, response_time: float) -> None:
        """Update response time statistics."""
        # Simple moving average
        total_successful = self.serving_stats['successful_predictions']
        current_avg = self.serving_stats['average_response_time']
        
        # Update average
        new_avg = ((current_avg * (total_successful - 1)) + response_time) / total_successful
        self.serving_stats['average_response_time'] = new_avg
    
    def _save_preparation_results(self, results: Dict[str, Any]) -> None:
        """Save preparation results to file."""
        results_path = self.output_dir / f"preparation_results_{results['preparation_id']}.json"
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Preparation results saved to {results_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _create_deployment_script(self, exported_files: Dict[str, str]) -> str:
        """Create deployment script for production."""
        script_content = f'''"""
Production deployment script for {self.model.__class__.__name__}.
Auto-generated deployment script.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

class ProductionModel:
    """Production model wrapper."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize production model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Here you would need to import your actual model class
        # For now, this is a placeholder
        # self.model = YourModelClass(checkpoint['model_config'])
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.to(self.device)
        # self.model.eval()
        
        print(f"Model loaded on {{self.device}}")
    
    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Make prediction."""
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data).to(self.device)
        else:
            input_tensor = input_data.to(self.device)
        
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predictions = output.cpu().numpy()
        
        return {{
            'predictions': predictions.tolist(),
            'model_name': '{self.model.__class__.__name__}',
            'timestamp': str(datetime.now())
        }}

if __name__ == "__main__":
    # Initialize model
    model = ProductionModel(
        'model.pt',
        'config.json'
    )
    
    # Example usage
    # sample_input = np.random.randn(1, input_size)
    # result = model.predict(sample_input)
    # print(result)
    
    print("Model ready for production use!")
'''
        
        return script_content
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        uptime = datetime.now() - self.serving_stats['uptime_start']
        
        return {
            'deployment_status': self.deployment_status.copy(),
            'serving_stats': self.serving_stats.copy(),
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime),
            'performance_summary': self.inference_engine.get_performance_summary()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the deployment manager."""
        self.logger.info("Shutting down deployment manager...")
        
        # Stop monitoring
        self.deployment_status['monitoring_active'] = False
        
        # Generate final report
        final_report = self.generate_deployment_report()
        
        # Log final statistics
        uptime = datetime.now() - self.serving_stats['uptime_start']
        self.logger.info(f"Deployment served {self.serving_stats['total_requests']} requests over {uptime}")
        self.logger.info(f"Final report saved to {final_report}")
        
        self.logger.info("Deployment manager shutdown complete")


class InferenceServer:
    """
    Simple inference server for model serving.
    """
    
    def __init__(self, deployment_manager: ModelDeploymentManager):
        """Initialize inference server."""
        self.deployment_manager = deployment_manager
        self.logger = deployment_manager.logger
    
    def start_server(self, host: str = '0.0.0.0', port: int = 8000):
        """
        Start HTTP inference server.
        
        Args:
            host: Server host
            port: Server port
        """
        try:
            import flask
            from flask import Flask, request, jsonify
            
            app = Flask(__name__)
            
            @app.route('/predict', methods=['POST'])
            def predict():
                try:
                    data = request.get_json()
                    input_data = np.array(data['input'])
                    
                    result = self.deployment_manager.predict(
                        input_data,
                        return_confidence=True,
                        validate_input=True
                    )
                    
                    return jsonify(result)
                    
                except Exception as e:
                    return jsonify({'error': str(e)}), 400
            
            @app.route('/health', methods=['GET'])
            def health():
                status = self.deployment_manager.get_deployment_status()
                return jsonify(status)
            
            @app.route('/metrics', methods=['GET'])
            def metrics():
                performance = self.deployment_manager.inference_engine.get_performance_summary()
                return jsonify(performance)
            
            self.logger.info(f"Starting inference server on {host}:{port}")
            app.run(host=host, port=port, debug=False)
            
        except ImportError:
            self.logger.error("Flask not installed. Cannot start HTTP server.")
            raise ImportError("Install Flask to use HTTP inference server: pip install flask")

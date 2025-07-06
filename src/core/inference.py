"""
Advanced inference framework for model deployment and real-time prediction.
Provides comprehensive inference capabilities with monitoring and validation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import json
import time
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base_model import BaseModel
from .evaluator import Evaluator
from ..visualization.inference_viz import InferenceVisualizer
from ..utils.logging import setup_logging


class InferenceEngine:
    """
    Advanced inference engine for model deployment and real-time prediction.
    """
    
    def __init__(self, 
                 model: BaseModel,
                 config: Optional[Dict[str, Any]] = None,
                 output_dir: str = 'results/inference'):
        """
        Initialize the inference engine.
        
        Args:
            model: Trained model for inference
            config: Inference configuration
            output_dir: Directory for saving inference results
        """
        self.model = model
        self.config = config or {}
        self.device = model.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging('inference_engine', self.output_dir / 'inference.log')
        
        # Initialize components
        self.evaluator = Evaluator(model, config)
        self.visualizer = InferenceVisualizer(str(self.output_dir))
        
        # Performance tracking
        self.performance_metrics = {
            'total_inferences': 0,
            'total_time': 0.0,
            'inference_times': [],
            'batch_sizes': [],
            'throughput_history': []
        }
        
        # Validation thresholds
        self.validation_config = self.config.get('validation', {})
        self.confidence_threshold = self.validation_config.get('confidence_threshold', 0.8)
        self.drift_threshold = self.validation_config.get('drift_threshold', 0.1)
        
        # Reference statistics for drift detection
        self.reference_stats = None
        
        self.logger.info(f"Inference engine initialized with model: {model.__class__.__name__}")
    
    def run_inference(self, 
                     input_data: Union[torch.Tensor, np.ndarray, pd.DataFrame, DataLoader],
                     return_confidence: bool = False,
                     return_features: bool = False,
                     validate_inputs: bool = True) -> Dict[str, Any]:
        """
        Run inference on input data with comprehensive analysis.
        
        Args:
            input_data: Input data for inference
            return_confidence: Whether to return confidence scores
            return_features: Whether to return feature analysis
            validate_inputs: Whether to validate input data
            
        Returns:
            Comprehensive inference results
        """
        start_time = time.time()
        
        self.logger.info("Starting inference...")
        
        # Prepare input data
        data_loader = self._prepare_input_data(input_data)
        
        # Validate inputs if requested
        if validate_inputs:
            validation_results = self._validate_inputs(data_loader)
            if not validation_results['valid']:
                self.logger.warning(f"Input validation failed: {validation_results['issues']}")
        
        # Run inference
        predictions, confidence_scores, feature_importance = self._run_model_inference(
            data_loader, return_confidence, return_features
        )
        
        # Calculate performance metrics
        inference_time = time.time() - start_time
        self._update_performance_metrics(inference_time, len(predictions))
        
        # Prepare results
        results = {
            'predictions': predictions,
            'num_samples': len(predictions),
            'inference_time': inference_time,
            'throughput': len(predictions) / inference_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_confidence:
            results['confidence_scores'] = confidence_scores
            results['avg_confidence'] = np.mean(confidence_scores) if confidence_scores is not None else None
        
        if return_features:
            results['feature_importance'] = feature_importance
        
        if validate_inputs:
            results['input_validation'] = validation_results
        
        # Detect data drift
        drift_analysis = self._detect_data_drift(data_loader)
        results['drift_analysis'] = drift_analysis
        
        self.logger.info(f"Inference completed: {len(predictions)} predictions in {inference_time:.3f}s")
        
        return results
    
    def batch_inference(self,
                       data_batches: List[Union[torch.Tensor, np.ndarray]],
                       parallel: bool = True,
                       max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Run inference on multiple batches of data.
        
        Args:
            data_batches: List of data batches
            parallel: Whether to run batches in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of inference results for each batch
        """
        self.logger.info(f"Starting batch inference on {len(data_batches)} batches")
        
        if parallel and len(data_batches) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.run_inference, batch, False, False, True)
                    for batch in data_batches
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                self.run_inference(batch, False, False, True)
                for batch in data_batches
            ]
        
        # Aggregate results
        aggregated_results = self._aggregate_batch_results(results)
        
        self.logger.info(f"Batch inference completed: {len(results)} batches processed")
        
        return results, aggregated_results
    
    def stream_inference(self,
                        data_stream: Callable,
                        window_size: int = 100,
                        update_interval: float = 1.0) -> None:
        """
        Run continuous inference on streaming data.
        
        Args:
            data_stream: Function that yields data points
            window_size: Size of sliding window for analysis
            update_interval: Time interval between updates (seconds)
        """
        self.logger.info("Starting streaming inference...")
        
        predictions_window = []
        confidence_window = []
        
        try:
            while True:
                # Get next data point
                data_point = next(data_stream())
                
                # Run inference
                result = self.run_inference(data_point, return_confidence=True)
                
                # Update sliding windows
                predictions_window.append(result['predictions'][0])
                if result.get('confidence_scores'):
                    confidence_window.append(result['confidence_scores'][0])
                
                # Maintain window size
                if len(predictions_window) > window_size:
                    predictions_window.pop(0)
                    if confidence_window:
                        confidence_window.pop(0)
                
                # Analyze trends
                if len(predictions_window) >= 10:  # Minimum points for analysis
                    trend_analysis = self._analyze_prediction_trends(
                        predictions_window, confidence_window
                    )
                    
                    # Log alerts if needed
                    if trend_analysis.get('anomaly_detected'):
                        self.logger.warning(f"Anomaly detected: {trend_analysis['anomaly_reason']}")
                
                # Wait for next update
                time.sleep(update_interval)
                
        except StopIteration:
            self.logger.info("Stream inference completed")
        except KeyboardInterrupt:
            self.logger.info("Stream inference interrupted by user")
    
    def validate_model_performance(self,
                                 test_data: DataLoader,
                                 baseline_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Validate current model performance against baseline or historical data.
        
        Args:
            test_data: Test dataset for validation
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            Performance validation results
        """
        self.logger.info("Starting model performance validation...")
        
        # Run evaluation
        evaluation_results = self.evaluator.evaluate(test_data, save_results=False)
        current_metrics = evaluation_results['metrics']
        
        # Performance validation
        validation_results = {
            'current_metrics': current_metrics,
            'validation_timestamp': datetime.now().isoformat(),
            'performance_status': 'UNKNOWN'
        }
        
        if baseline_metrics:
            # Compare with baseline
            comparison = self._compare_metrics(current_metrics, baseline_metrics)
            validation_results['baseline_comparison'] = comparison
            validation_results['performance_degradation'] = comparison.get('degradation', {})
            
            # Determine overall status
            degradation_threshold = self.validation_config.get('degradation_threshold', 0.05)
            significant_degradation = any(
                abs(deg) > degradation_threshold 
                for deg in comparison.get('degradation', {}).values()
                if isinstance(deg, (int, float))
            )
            
            if significant_degradation:
                validation_results['performance_status'] = 'DEGRADED'
                self.logger.warning("Model performance degradation detected")
            else:
                validation_results['performance_status'] = 'STABLE'
                self.logger.info("Model performance is stable")
        
        # Benchmark inference speed
        benchmark_results = self.evaluator.benchmark_inference(test_data)
        validation_results['inference_benchmark'] = benchmark_results
        
        # Generate validation report
        self._generate_validation_report(validation_results)
        
        return validation_results
    
    def set_reference_statistics(self, reference_data: DataLoader) -> None:
        """
        Set reference statistics for data drift detection.
        
        Args:
            reference_data: Reference dataset (typically training data)
        """
        self.logger.info("Computing reference statistics for drift detection...")
        
        all_data = []
        for data, _ in reference_data:
            all_data.append(data.numpy())
        
        all_data = np.concatenate(all_data, axis=0)
        
        self.reference_stats = {
            'mean': np.mean(all_data, axis=0),
            'std': np.std(all_data, axis=0),
            'min': np.min(all_data, axis=0),
            'max': np.max(all_data, axis=0),
            'quantiles': np.percentile(all_data, [25, 50, 75], axis=0)
        }
        
        self.logger.info("Reference statistics computed successfully")
    
    def export_inference_model(self, 
                              export_path: str,
                              format: str = 'torch',
                              include_metadata: bool = True) -> str:
        """
        Export model for deployment.
        
        Args:
            export_path: Path to save the exported model
            format: Export format ('torch', 'onnx', 'torchscript')
            include_metadata: Whether to include metadata
            
        Returns:
            Path to exported model
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting model in {format} format to {export_path}")
        
        if format == 'torch':
            model_path = export_path / 'model.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': self.model.config,
                'model_class': self.model.__class__.__name__
            }, model_path)
        
        elif format == 'torchscript':
            model_path = export_path / 'model_scripted.pt'
            scripted_model = torch.jit.script(self.model)
            torch.jit.save(scripted_model, model_path)
        
        elif format == 'onnx':
            try:
                import torch.onnx
                model_path = export_path / 'model.onnx'
                
                # Create dummy input
                dummy_input = torch.randn(1, *self.model.input_shape).to(self.device)
                
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    model_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
            except ImportError:
                raise ImportError("ONNX export requires 'onnx' package")
        
        if include_metadata:
            metadata = {
                'model_name': self.model.__class__.__name__,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'num_parameters': self.model.count_parameters(),
                'export_timestamp': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics
            }
            
            with open(export_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model exported successfully to {model_path}")
        return str(model_path)
    
    def _prepare_input_data(self, input_data: Union[torch.Tensor, np.ndarray, pd.DataFrame, DataLoader]) -> DataLoader:
        """Prepare input data for inference."""
        if isinstance(input_data, DataLoader):
            return input_data
        
        # Convert to tensor
        if isinstance(input_data, pd.DataFrame):
            data_tensor = torch.FloatTensor(input_data.values)
        elif isinstance(input_data, np.ndarray):
            data_tensor = torch.FloatTensor(input_data)
        elif isinstance(input_data, torch.Tensor):
            data_tensor = input_data.float()
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
        
        # Ensure proper shape
        if data_tensor.dim() == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        # Create dummy targets for DataLoader compatibility
        dummy_targets = torch.zeros(data_tensor.shape[0])
        dataset = TensorDataset(data_tensor, dummy_targets)
        
        batch_size = self.config.get('inference_batch_size', 32)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def _run_model_inference(self, 
                           data_loader: DataLoader,
                           return_confidence: bool,
                           return_features: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Run model inference with optional confidence and feature analysis."""
        self.model.eval()
        
        all_predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                # Get predictions
                if self.config.get('task_type') == 'classification':
                    predictions = output.argmax(dim=1)
                    if return_confidence:
                        confidences = torch.softmax(output, dim=1).max(dim=1)[0]
                        all_confidences.extend(confidences.cpu().numpy())
                else:
                    predictions = output.squeeze()
                    if return_confidence:
                        # For regression, confidence could be based on uncertainty estimation
                        # This is a simple placeholder - more sophisticated methods exist
                        confidences = torch.ones_like(predictions) * 0.9
                        all_confidences.extend(confidences.cpu().numpy())
                
                all_predictions.extend(predictions.cpu().numpy())
        
        predictions = np.array(all_predictions)
        confidence_scores = np.array(all_confidences) if all_confidences else None
        
        # Feature importance (placeholder for gradient-based methods)
        feature_importance = None
        if return_features:
            try:
                feature_importance = self.evaluator.get_feature_importance(data_loader)
            except Exception as e:
                self.logger.warning(f"Feature importance calculation failed: {e}")
        
        return predictions, confidence_scores, feature_importance
    
    def _validate_inputs(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Validate input data quality and format."""
        issues = []
        
        sample_data, _ = next(iter(data_loader))
        
        # Check for NaN values
        if torch.isnan(sample_data).any():
            issues.append("NaN values detected in input data")
        
        # Check for infinite values
        if torch.isinf(sample_data).any():
            issues.append("Infinite values detected in input data")
        
        # Check input shape
        expected_shape = self.model.input_shape
        if sample_data.shape[1:] != torch.Size(expected_shape):
            issues.append(f"Input shape mismatch: expected {expected_shape}, got {sample_data.shape[1:]}")
        
        # Check value ranges (if reference stats available)
        if self.reference_stats is not None:
            data_np = sample_data.numpy()
            data_mean = np.mean(data_np, axis=0)
            data_std = np.std(data_np, axis=0)
            
            # Check for significant distribution shifts
            mean_shift = np.abs(data_mean - self.reference_stats['mean']) / self.reference_stats['std']
            if np.any(mean_shift > 3):  # 3-sigma rule
                issues.append("Significant distribution shift detected")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'num_samples': len(data_loader.dataset)
        }
    
    def _detect_data_drift(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Detect data drift compared to reference statistics."""
        if self.reference_stats is None:
            return {'drift_detected': False, 'reason': 'No reference statistics available'}
        
        # Compute current data statistics
        all_data = []
        for data, _ in data_loader:
            all_data.append(data.numpy())
        
        current_data = np.concatenate(all_data, axis=0)
        current_stats = {
            'mean': np.mean(current_data, axis=0),
            'std': np.std(current_data, axis=0)
        }
        
        # Calculate drift metrics
        mean_drift = np.mean(np.abs(current_stats['mean'] - self.reference_stats['mean']))
        std_drift = np.mean(np.abs(current_stats['std'] - self.reference_stats['std']))
        
        drift_detected = mean_drift > self.drift_threshold or std_drift > self.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'mean_drift': float(mean_drift),
            'std_drift': float(std_drift),
            'drift_threshold': self.drift_threshold,
            'drift_score': float(max(mean_drift, std_drift))
        }
    
    def _update_performance_metrics(self, inference_time: float, num_samples: int) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics['total_inferences'] += num_samples
        self.performance_metrics['total_time'] += inference_time
        self.performance_metrics['inference_times'].append(inference_time)
        self.performance_metrics['batch_sizes'].append(num_samples)
        
        # Calculate throughput
        throughput = num_samples / inference_time
        self.performance_metrics['throughput_history'].append(throughput)
        
        # Keep only recent history (last 100 entries)
        for key in ['inference_times', 'batch_sizes', 'throughput_history']:
            if len(self.performance_metrics[key]) > 100:
                self.performance_metrics[key] = self.performance_metrics[key][-100:]
    
    def _analyze_prediction_trends(self, 
                                 predictions: List[float],
                                 confidences: List[float]) -> Dict[str, Any]:
        """Analyze trends in predictions for anomaly detection."""
        predictions_arr = np.array(predictions)
        
        # Calculate trend metrics
        recent_mean = np.mean(predictions_arr[-10:])
        overall_mean = np.mean(predictions_arr)
        trend_deviation = abs(recent_mean - overall_mean) / (np.std(predictions_arr) + 1e-8)
        
        # Detect anomalies
        anomaly_detected = False
        anomaly_reason = None
        
        if trend_deviation > 2.0:  # 2-sigma threshold
            anomaly_detected = True
            anomaly_reason = "Significant trend deviation detected"
        
        if confidences:
            recent_confidence = np.mean(confidences[-10:])
            if recent_confidence < self.confidence_threshold:
                anomaly_detected = True
                anomaly_reason = f"Low confidence detected: {recent_confidence:.3f}"
        
        return {
            'anomaly_detected': anomaly_detected,
            'anomaly_reason': anomaly_reason,
            'trend_deviation': float(trend_deviation),
            'recent_mean': float(recent_mean),
            'overall_mean': float(overall_mean)
        }
    
    def _compare_metrics(self, 
                        current_metrics: Dict[str, float],
                        baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {
            'degradation': {},
            'improvement': {},
            'stable': {}
        }
        
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics and isinstance(current_value, (int, float)):
                baseline_value = baseline_metrics[metric]
                
                if isinstance(baseline_value, (int, float)):
                    relative_change = (current_value - baseline_value) / (abs(baseline_value) + 1e-8)
                    
                    if abs(relative_change) < 0.01:  # 1% threshold
                        comparison['stable'][metric] = relative_change
                    elif relative_change < 0:  # Performance degradation (assuming higher is better)
                        comparison['degradation'][metric] = relative_change
                    else:  # Performance improvement
                        comparison['improvement'][metric] = relative_change
        
        return comparison
    
    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple batches."""
        total_samples = sum(result['num_samples'] for result in batch_results)
        total_time = sum(result['inference_time'] for result in batch_results)
        
        # Combine predictions
        all_predictions = []
        for result in batch_results:
            all_predictions.extend(result['predictions'])
        
        return {
            'total_samples': total_samples,
            'total_time': total_time,
            'average_throughput': total_samples / total_time,
            'num_batches': len(batch_results),
            'all_predictions': all_predictions,
            'prediction_stats': {
                'mean': np.mean(all_predictions),
                'std': np.std(all_predictions),
                'min': np.min(all_predictions),
                'max': np.max(all_predictions)
            }
        }
    
    def _generate_validation_report(self, validation_results: Dict[str, Any]) -> None:
        """Generate a comprehensive validation report."""
        report_content = f"""
# Model Performance Validation Report

## Validation Summary
- **Timestamp**: {validation_results['validation_timestamp']}
- **Performance Status**: {validation_results['performance_status']}

## Current Metrics
"""
        
        for metric, value in validation_results['current_metrics'].items():
            if isinstance(value, (int, float)):
                report_content += f"- **{metric}**: {value:.4f}\n"
        
        if 'baseline_comparison' in validation_results:
            report_content += f"""
## Baseline Comparison
### Performance Changes
"""
            comparison = validation_results['baseline_comparison']
            
            for category, metrics in comparison.items():
                if metrics:
                    report_content += f"\n**{category.title()}**:\n"
                    for metric, change in metrics.items():
                        report_content += f"- {metric}: {change:+.2%}\n"
        
        if 'inference_benchmark' in validation_results:
            benchmark = validation_results['inference_benchmark']
            report_content += f"""
## Inference Performance
- **Mean Inference Time**: {benchmark.get('mean_inference_time', 0):.4f}s
- **Std Inference Time**: {benchmark.get('std_inference_time', 0):.4f}s
"""
        
        # Save report
        report_path = self.output_dir / 'validation_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Validation report saved to {report_path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if self.performance_metrics['total_inferences'] == 0:
            return {'message': 'No inference operations performed yet'}
        
        avg_inference_time = self.performance_metrics['total_time'] / len(self.performance_metrics['inference_times'])
        avg_throughput = np.mean(self.performance_metrics['throughput_history'])
        
        return {
            'total_inferences': self.performance_metrics['total_inferences'],
            'total_time': self.performance_metrics['total_time'],
            'average_inference_time': avg_inference_time,
            'average_throughput': avg_throughput,
            'inference_operations': len(self.performance_metrics['inference_times']),
            'throughput_trend': {
                'recent_avg': np.mean(self.performance_metrics['throughput_history'][-10:]) if len(self.performance_metrics['throughput_history']) >= 10 else avg_throughput,
                'overall_avg': avg_throughput
            }
        }

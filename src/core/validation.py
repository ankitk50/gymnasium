"""
Advanced model validation framework for comprehensive performance assessment.
Provides automated validation pipelines and continuous monitoring capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import json
import time
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report
)

from .base_model import BaseModel
from .evaluator import Evaluator
from .inference import InferenceEngine
from ..visualization.evaluation_viz import EvaluationVisualizer
from ..utils.logging import setup_logging


class ModelValidator:
    """
    Comprehensive model validation framework with automated testing pipelines.
    """
    
    def __init__(self, 
                 model: BaseModel,
                 config: Optional[Dict[str, Any]] = None,
                 output_dir: str = 'validation_output'):
        """
        Initialize the model validator.
        
        Args:
            model: Model to validate
            config: Validation configuration
            output_dir: Directory for saving validation results
        """
        self.model = model
        self.config = config or {}
        self.device = model.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging('model_validator', self.output_dir / 'validation.log')
        
        # Initialize components
        self.evaluator = Evaluator(model, config)
        self.inference_engine = InferenceEngine(model, config, str(self.output_dir / 'inference'))
        self.visualizer = EvaluationVisualizer(str(self.output_dir))
        
        # Validation thresholds and criteria
        self.validation_criteria = self._setup_validation_criteria()
        
        # Historical results storage
        self.validation_history = []
        
        self.logger.info(f"Model validator initialized for {model.__class__.__name__}")
    
    def comprehensive_validation(self,
                               train_data: DataLoader,
                               val_data: DataLoader,
                               test_data: DataLoader,
                               baseline_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Run comprehensive model validation across multiple dimensions.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset 
            test_data: Test dataset
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("Starting comprehensive model validation...")
        
        validation_results = {
            'validation_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'model_info': self._get_model_info(),
            'validation_summary': {}
        }
        
        # 1. Basic Performance Evaluation
        self.logger.info("1. Running basic performance evaluation...")
        basic_eval = self.evaluator.evaluate(test_data, save_results=False)
        validation_results['basic_evaluation'] = basic_eval
        
        # 2. Cross-Validation
        self.logger.info("2. Running cross-validation...")
        cv_results = self._run_cross_validation(train_data)
        validation_results['cross_validation'] = cv_results
        
        # 3. Robustness Testing
        self.logger.info("3. Running robustness tests...")
        robustness_results = self._test_robustness(test_data)
        validation_results['robustness'] = robustness_results
        
        # 4. Generalization Analysis
        self.logger.info("4. Analyzing generalization...")
        generalization_results = self._analyze_generalization(train_data, val_data, test_data)
        validation_results['generalization'] = generalization_results
        
        # 5. Performance Consistency
        self.logger.info("5. Testing performance consistency...")
        consistency_results = self._test_consistency(test_data)
        validation_results['consistency'] = consistency_results
        
        # 6. Inference Performance
        self.logger.info("6. Benchmarking inference performance...")
        inference_benchmark = self._benchmark_inference(test_data)
        validation_results['inference_performance'] = inference_benchmark
        
        # 7. Data Sensitivity Analysis
        self.logger.info("7. Running data sensitivity analysis...")
        sensitivity_results = self._analyze_data_sensitivity(test_data)
        validation_results['data_sensitivity'] = sensitivity_results
        
        # 8. Baseline Comparison (if provided)
        if baseline_metrics:
            self.logger.info("8. Comparing with baseline...")
            baseline_comparison = self._compare_with_baseline(
                basic_eval['metrics'], baseline_metrics
            )
            validation_results['baseline_comparison'] = baseline_comparison
        
        # 9. Generate Overall Assessment
        overall_assessment = self._generate_overall_assessment(validation_results)
        validation_results['validation_summary'] = overall_assessment
        
        # 10. Save Results and Generate Reports
        self._save_validation_results(validation_results)
        self._generate_validation_visualizations(validation_results)
        
        # Add to history
        self.validation_history.append(validation_results)
        
        self.logger.info("Comprehensive validation completed successfully!")
        
        return validation_results
    
    def continuous_validation(self,
                            test_data: DataLoader,
                            monitoring_interval: int = 3600,  # 1 hour
                            max_duration: int = 86400) -> None:  # 24 hours
        """
        Run continuous validation monitoring.
        
        Args:
            test_data: Test dataset for validation
            monitoring_interval: Interval between validations (seconds)
            max_duration: Maximum monitoring duration (seconds)
        """
        self.logger.info(f"Starting continuous validation monitoring for {max_duration/3600:.1f} hours")
        
        start_time = time.time()
        validation_count = 0
        
        try:
            while time.time() - start_time < max_duration:
                validation_count += 1
                self.logger.info(f"Running validation cycle {validation_count}")
                
                # Run quick validation
                quick_results = self._quick_validation(test_data)
                
                # Check for performance degradation
                if self._detect_performance_degradation(quick_results):
                    self.logger.warning("Performance degradation detected!")
                    
                    # Run comprehensive validation
                    comprehensive_results = self.comprehensive_validation(
                        test_data, test_data, test_data  # Using test_data for all sets in monitoring
                    )
                    
                    # Alert if critical issues found
                    if comprehensive_results['validation_summary']['overall_status'] == 'FAILED':
                        self.logger.error("Critical validation failure detected!")
                        break
                
                # Wait for next cycle
                time.sleep(monitoring_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Continuous validation interrupted by user")
        
        self.logger.info(f"Continuous validation completed after {validation_count} cycles")
    
    def validate_data_quality(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Validate data quality and detect potential issues.
        
        Args:
            data_loader: Data to validate
            
        Returns:
            Data quality assessment results
        """
        self.logger.info("Validating data quality...")
        
        issues = []
        warnings = []
        statistics = {}
        
        all_data = []
        all_targets = []
        
        # Collect all data
        for data, targets in data_loader:
            all_data.append(data.numpy())
            all_targets.append(targets.numpy())
        
        all_data = np.concatenate(all_data, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Basic statistics
        statistics['num_samples'] = len(all_data)
        statistics['num_features'] = all_data.shape[1] if len(all_data.shape) > 1 else 1
        statistics['data_shape'] = all_data.shape
        
        # Check for missing values
        nan_count = np.isnan(all_data).sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values in data")
        
        # Check for infinite values
        inf_count = np.isinf(all_data).sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values in data")
        
        # Check for constant features
        if len(all_data.shape) > 1:
            constant_features = []
            for i in range(all_data.shape[1]):
                if np.std(all_data[:, i]) < 1e-8:
                    constant_features.append(i)
            
            if constant_features:
                warnings.append(f"Found {len(constant_features)} constant features: {constant_features}")
        
        # Check data distribution
        statistics['data_stats'] = {
            'mean': np.mean(all_data, axis=0).tolist(),
            'std': np.std(all_data, axis=0).tolist(),
            'min': np.min(all_data, axis=0).tolist(),
            'max': np.max(all_data, axis=0).tolist()
        }
        
        # Check target distribution
        statistics['target_stats'] = {
            'mean': float(np.mean(all_targets)),
            'std': float(np.std(all_targets)),
            'min': float(np.min(all_targets)),
            'max': float(np.max(all_targets)),
            'unique_values': len(np.unique(all_targets))
        }
        
        # Check for class imbalance (classification)
        if self.config.get('task_type') == 'classification':
            unique_targets, counts = np.unique(all_targets, return_counts=True)
            class_distribution = dict(zip(unique_targets.astype(int), counts))
            statistics['class_distribution'] = class_distribution
            
            # Check for severe imbalance
            min_class_ratio = min(counts) / len(all_targets)
            if min_class_ratio < 0.05:  # Less than 5%
                warnings.append(f"Severe class imbalance detected. Smallest class: {min_class_ratio:.2%}")
        
        # Overall quality assessment
        quality_score = self._calculate_data_quality_score(issues, warnings, statistics)
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'warnings': warnings,
            'statistics': statistics,
            'recommendations': self._generate_data_quality_recommendations(issues, warnings)
        }
    
    def validate_model_interpretability(self, test_data: DataLoader) -> Dict[str, Any]:
        """
        Validate model interpretability and explainability.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Interpretability assessment results
        """
        self.logger.info("Validating model interpretability...")
        
        interpretability_results = {
            'feature_importance_available': False,
            'prediction_explainability': 'low',
            'model_complexity': self._assess_model_complexity(),
            'recommendations': []
        }
        
        # Try to get feature importance
        try:
            feature_importance = self.evaluator.get_feature_importance(test_data)
            if feature_importance:
                interpretability_results['feature_importance_available'] = True
                interpretability_results['feature_importance'] = feature_importance
                interpretability_results['prediction_explainability'] = 'medium'
        except Exception as e:
            self.logger.warning(f"Could not compute feature importance: {e}")
            interpretability_results['recommendations'].append(
                "Consider using model-agnostic interpretability methods"
            )
        
        # Assess model architecture complexity
        complexity = interpretability_results['model_complexity']
        if complexity['num_parameters'] > 1e6:
            interpretability_results['recommendations'].append(
                "Large model with limited interpretability - consider model distillation"
            )
        
        if complexity['num_layers'] > 10:
            interpretability_results['recommendations'].append(
                "Deep model - consider attention visualization or layer-wise relevance propagation"
            )
        
        return interpretability_results
    
    def benchmark_against_baselines(self,
                                  test_data: DataLoader,
                                  baseline_models: Optional[List[BaseModel]] = None) -> Dict[str, Any]:
        """
        Benchmark model against baseline implementations.
        
        Args:
            test_data: Test dataset
            baseline_models: List of baseline models to compare against
            
        Returns:
            Benchmark comparison results
        """
        self.logger.info("Benchmarking against baseline models...")
        
        benchmark_results = {
            'main_model': {
                'name': self.model.__class__.__name__,
                'metrics': self.evaluator.evaluate(test_data, save_results=False)['metrics']
            },
            'baselines': {},
            'comparison': {}
        }
        
        if baseline_models:
            for i, baseline_model in enumerate(baseline_models):
                baseline_name = f"baseline_{i}_{baseline_model.__class__.__name__}"
                
                # Evaluate baseline
                baseline_evaluator = Evaluator(baseline_model, self.config)
                baseline_metrics = baseline_evaluator.evaluate(test_data, save_results=False)['metrics']
                
                benchmark_results['baselines'][baseline_name] = {
                    'name': baseline_model.__class__.__name__,
                    'metrics': baseline_metrics
                }
        
        # Create comparison
        main_metrics = benchmark_results['main_model']['metrics']
        for baseline_name, baseline_info in benchmark_results['baselines'].items():
            baseline_metrics = baseline_info['metrics']
            
            comparison = {}
            for metric, main_value in main_metrics.items():
                if metric in baseline_metrics and isinstance(main_value, (int, float)):
                    baseline_value = baseline_metrics[metric]
                    if isinstance(baseline_value, (int, float)):
                        improvement = (main_value - baseline_value) / (abs(baseline_value) + 1e-8)
                        comparison[metric] = {
                            'main_value': main_value,
                            'baseline_value': baseline_value,
                            'improvement': improvement,
                            'improvement_pct': improvement * 100
                        }
            
            benchmark_results['comparison'][baseline_name] = comparison
        
        return benchmark_results
    
    def _setup_validation_criteria(self) -> Dict[str, Any]:
        """Setup validation criteria and thresholds."""
        default_criteria = {
            'minimum_accuracy': 0.7,
            'minimum_r2_score': 0.6,
            'maximum_mse': 1.0,
            'minimum_f1_score': 0.7,
            'maximum_inference_time': 1.0,  # seconds
            'minimum_consistency_score': 0.8,
            'maximum_drift_threshold': 0.1
        }
        
        # Override with config values
        criteria = {**default_criteria, **self.config.get('validation_criteria', {})}
        
        return criteria
    
    def _run_cross_validation(self, train_data: DataLoader, k_folds: int = 5) -> Dict[str, Any]:
        """Run k-fold cross-validation."""
        try:
            cv_results = self.evaluator.cross_validate(train_data, k_folds)
            return {
                'status': 'success',
                'results': cv_results,
                'mean_performance': cv_results.get('cv_metrics', {}),
                'stability_assessment': self._assess_cv_stability(cv_results)
            }
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_robustness(self, test_data: DataLoader) -> Dict[str, Any]:
        """Test model robustness with various perturbations."""
        robustness_results = {
            'noise_robustness': {},
            'adversarial_robustness': {},
            'data_corruption_robustness': {}
        }
        
        # Test with different noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        for noise_level in noise_levels:
            noisy_results = self._evaluate_with_noise(test_data, noise_level)
            robustness_results['noise_robustness'][f'noise_{noise_level}'] = noisy_results
        
        # Test with missing features
        missing_feature_results = self._evaluate_with_missing_features(test_data)
        robustness_results['data_corruption_robustness'] = missing_feature_results
        
        return robustness_results
    
    def _analyze_generalization(self,
                              train_data: DataLoader,
                              val_data: DataLoader,
                              test_data: DataLoader) -> Dict[str, Any]:
        """Analyze model generalization capabilities."""
        
        # Evaluate on all datasets
        train_metrics = self.evaluator.evaluate(train_data, save_results=False)['metrics']
        val_metrics = self.evaluator.evaluate(val_data, save_results=False)['metrics']
        test_metrics = self.evaluator.evaluate(test_data, save_results=False)['metrics']
        
        # Calculate generalization gaps
        generalization_analysis = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'generalization_gaps': {},
            'overfitting_assessment': {}
        }
        
        # Calculate gaps
        for metric in train_metrics:
            if metric in val_metrics and isinstance(train_metrics[metric], (int, float)):
                train_val_gap = train_metrics[metric] - val_metrics[metric]
                generalization_analysis['generalization_gaps'][f'{metric}_train_val_gap'] = train_val_gap
                
                if metric in test_metrics:
                    train_test_gap = train_metrics[metric] - test_metrics[metric]
                    generalization_analysis['generalization_gaps'][f'{metric}_train_test_gap'] = train_test_gap
        
        # Overfitting assessment
        if 'test_loss' in train_metrics and 'test_loss' in test_metrics:
            loss_gap = test_metrics['test_loss'] - train_metrics['test_loss']
            if loss_gap > 0.1:  # 10% threshold
                generalization_analysis['overfitting_assessment']['status'] = 'potential_overfitting'
                generalization_analysis['overfitting_assessment']['loss_gap'] = loss_gap
            else:
                generalization_analysis['overfitting_assessment']['status'] = 'good_generalization'
        
        return generalization_analysis
    
    def _test_consistency(self, test_data: DataLoader, num_runs: int = 5) -> Dict[str, Any]:
        """Test prediction consistency across multiple runs."""
        consistency_results = {
            'prediction_variance': {},
            'stability_score': 0.0,
            'recommendations': []
        }
        
        all_predictions = []
        all_metrics = []
        
        for run in range(num_runs):
            # Add slight randomness to test consistency
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for data, _ in test_data:
                    data = data.to(self.device)
                    output = self.model(data)
                    
                    if self.config.get('task_type') == 'classification':
                        pred = output.argmax(dim=1)
                    else:
                        pred = output.squeeze()
                    
                    predictions.extend(pred.cpu().numpy())
            
            all_predictions.append(predictions)
            
            # Evaluate metrics for this run
            metrics = self.evaluator.evaluate(test_data, save_results=False)['metrics']
            all_metrics.append(metrics)
        
        # Calculate consistency metrics
        predictions_array = np.array(all_predictions)
        prediction_variance = np.var(predictions_array, axis=0)
        
        consistency_results['prediction_variance'] = {
            'mean_variance': float(np.mean(prediction_variance)),
            'max_variance': float(np.max(prediction_variance)),
            'variance_distribution': prediction_variance.tolist()
        }
        
        # Calculate stability score based on metric consistency
        metric_variances = {}
        for metric in all_metrics[0]:
            if isinstance(all_metrics[0][metric], (int, float)):
                values = [run_metrics[metric] for run_metrics in all_metrics]
                metric_variances[metric] = np.var(values)
        
        # Stability score (lower variance = higher stability)
        avg_variance = np.mean(list(metric_variances.values()))
        stability_score = max(0, 1 - avg_variance)  # Normalized to [0, 1]
        
        consistency_results['stability_score'] = float(stability_score)
        consistency_results['metric_variances'] = metric_variances
        
        if stability_score < 0.8:
            consistency_results['recommendations'].append(
                "Low prediction consistency detected - consider ensemble methods or regularization"
            )
        
        return consistency_results
    
    def _benchmark_inference(self, test_data: DataLoader) -> Dict[str, Any]:
        """Benchmark inference performance."""
        return self.evaluator.benchmark_inference(test_data, num_runs=50)
    
    def _analyze_data_sensitivity(self, test_data: DataLoader) -> Dict[str, Any]:
        """Analyze model sensitivity to input data variations."""
        sensitivity_results = {
            'feature_sensitivity': {},
            'input_range_sensitivity': {},
            'recommendations': []
        }
        
        # Test sensitivity to individual feature perturbations
        sample_data, _ = next(iter(test_data))
        baseline_output = self.model(sample_data.to(self.device))
        
        if len(sample_data.shape) > 1:
            for feature_idx in range(sample_data.shape[1]):
                # Perturb each feature slightly
                perturbed_data = sample_data.clone()
                perturbed_data[:, feature_idx] += 0.1 * torch.std(sample_data[:, feature_idx])
                
                perturbed_output = self.model(perturbed_data.to(self.device))
                
                # Calculate sensitivity
                output_change = torch.mean(torch.abs(perturbed_output - baseline_output))
                sensitivity_results['feature_sensitivity'][f'feature_{feature_idx}'] = float(output_change)
        
        return sensitivity_results
    
    def _compare_with_baseline(self,
                             current_metrics: Dict[str, float],
                             baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {
            'improvements': {},
            'degradations': {},
            'overall_status': 'unknown'
        }
        
        significant_changes = 0
        improvements = 0
        degradations = 0
        
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics and isinstance(current_value, (int, float)):
                baseline_value = baseline_metrics[metric]
                if isinstance(baseline_value, (int, float)):
                    
                    relative_change = (current_value - baseline_value) / (abs(baseline_value) + 1e-8)
                    
                    if abs(relative_change) > 0.05:  # 5% threshold
                        significant_changes += 1
                        
                        if relative_change > 0:  # Assuming higher is better
                            improvements += 1
                            comparison['improvements'][metric] = {
                                'current': current_value,
                                'baseline': baseline_value,
                                'improvement': relative_change
                            }
                        else:
                            degradations += 1
                            comparison['degradations'][metric] = {
                                'current': current_value,
                                'baseline': baseline_value,
                                'degradation': relative_change
                            }
        
        # Determine overall status
        if degradations == 0 and improvements > 0:
            comparison['overall_status'] = 'improved'
        elif degradations > improvements:
            comparison['overall_status'] = 'degraded'
        elif significant_changes == 0:
            comparison['overall_status'] = 'stable'
        else:
            comparison['overall_status'] = 'mixed'
        
        return comparison
    
    def _generate_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation assessment."""
        
        assessment = {
            'overall_status': 'UNKNOWN',
            'passed_tests': [],
            'failed_tests': [],
            'warnings': [],
            'recommendations': [],
            'quality_score': 0.0
        }
        
        scores = []
        
        # Check basic evaluation
        if 'basic_evaluation' in validation_results:
            metrics = validation_results['basic_evaluation']['metrics']
            
            # Check against criteria
            if self.config.get('task_type') == 'classification':
                if 'accuracy' in metrics:
                    if metrics['accuracy'] >= self.validation_criteria['minimum_accuracy']:
                        assessment['passed_tests'].append('accuracy_threshold')
                        scores.append(1.0)
                    else:
                        assessment['failed_tests'].append('accuracy_threshold')
                        scores.append(0.0)
            else:
                if 'r2_score' in metrics:
                    if metrics['r2_score'] >= self.validation_criteria['minimum_r2_score']:
                        assessment['passed_tests'].append('r2_threshold')
                        scores.append(1.0)
                    else:
                        assessment['failed_tests'].append('r2_threshold')
                        scores.append(0.0)
        
        # Check cross-validation stability
        if 'cross_validation' in validation_results:
            cv_results = validation_results['cross_validation']
            if cv_results['status'] == 'success':
                assessment['passed_tests'].append('cross_validation')
                scores.append(1.0)
            else:
                assessment['failed_tests'].append('cross_validation')
                scores.append(0.0)
        
        # Check consistency
        if 'consistency' in validation_results:
            stability_score = validation_results['consistency']['stability_score']
            if stability_score >= self.validation_criteria['minimum_consistency_score']:
                assessment['passed_tests'].append('consistency_check')
                scores.append(stability_score)
            else:
                assessment['failed_tests'].append('consistency_check')
                scores.append(stability_score)
        
        # Check inference performance
        if 'inference_performance' in validation_results:
            inference_time = validation_results['inference_performance']['mean_inference_time']
            if inference_time <= self.validation_criteria['maximum_inference_time']:
                assessment['passed_tests'].append('inference_performance')
                scores.append(1.0)
            else:
                assessment['warnings'].append('inference_performance_slow')
                scores.append(0.5)
        
        # Calculate overall quality score
        if scores:
            assessment['quality_score'] = np.mean(scores)
        
        # Determine overall status
        if len(assessment['failed_tests']) == 0:
            if assessment['quality_score'] >= 0.8:
                assessment['overall_status'] = 'PASSED'
            else:
                assessment['overall_status'] = 'PASSED_WITH_WARNINGS'
        else:
            assessment['overall_status'] = 'FAILED'
        
        # Generate recommendations
        if assessment['overall_status'] == 'FAILED':
            assessment['recommendations'].append("Model validation failed - consider retraining or architecture changes")
        elif len(assessment['warnings']) > 0:
            assessment['recommendations'].append("Model passed but has warnings - monitor performance")
        
        return assessment
    
    def _quick_validation(self, test_data: DataLoader) -> Dict[str, Any]:
        """Run quick validation for continuous monitoring."""
        start_time = time.time()
        
        # Basic evaluation
        basic_results = self.evaluator.evaluate(test_data, save_results=False)
        
        validation_time = time.time() - start_time
        
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': basic_results['metrics'],
            'validation_time': validation_time,
            'status': 'completed'
        }
    
    def _detect_performance_degradation(self, current_results: Dict[str, Any]) -> bool:
        """Detect if performance has degraded compared to history."""
        if not self.validation_history:
            return False
        
        # Get last validation results
        last_validation = self.validation_history[-1]
        last_metrics = last_validation.get('basic_evaluation', {}).get('metrics', {})
        current_metrics = current_results.get('metrics', {})
        
        # Check for significant degradation
        for metric, current_value in current_metrics.items():
            if metric in last_metrics and isinstance(current_value, (int, float)):
                last_value = last_metrics[metric]
                if isinstance(last_value, (int, float)):
                    
                    # Calculate relative change
                    relative_change = (current_value - last_value) / (abs(last_value) + 1e-8)
                    
                    # Check for degradation (assuming higher is better for most metrics)
                    if metric == 'test_loss':  # Lower is better for loss
                        if relative_change > 0.1:  # 10% increase in loss
                            return True
                    else:  # Higher is better for other metrics
                        if relative_change < -0.1:  # 10% decrease
                            return True
        
        return False
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'name': self.model.__class__.__name__,
            'num_parameters': self.model.count_parameters(),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'device': str(self.device),
            'config': self.model.config
        }
    
    def _assess_cv_stability(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cross-validation stability."""
        cv_metrics = cv_results.get('cv_metrics', {})
        
        stability_assessment = {
            'stable_metrics': [],
            'unstable_metrics': [],
            'overall_stability': 'good'
        }
        
        for metric, value in cv_metrics.items():
            if metric.endswith('_std'):
                base_metric = metric.replace('_std', '')
                if base_metric + '_mean' in cv_metrics:
                    mean_value = cv_metrics[base_metric + '_mean']
                    std_value = value
                    
                    # Calculate coefficient of variation
                    cv_ratio = std_value / (abs(mean_value) + 1e-8)
                    
                    if cv_ratio < 0.1:  # 10% threshold
                        stability_assessment['stable_metrics'].append(base_metric)
                    else:
                        stability_assessment['unstable_metrics'].append(base_metric)
        
        if len(stability_assessment['unstable_metrics']) > len(stability_assessment['stable_metrics']):
            stability_assessment['overall_stability'] = 'poor'
        elif len(stability_assessment['unstable_metrics']) > 0:
            stability_assessment['overall_stability'] = 'moderate'
        
        return stability_assessment
    
    def _evaluate_with_noise(self, test_data: DataLoader, noise_level: float) -> Dict[str, Any]:
        """Evaluate model with added noise."""
        self.model.eval()
        noisy_predictions = []
        
        with torch.no_grad():
            for data, _ in test_data:
                # Add noise
                noise = torch.randn_like(data) * noise_level
                noisy_data = data + noise
                
                output = self.model(noisy_data.to(self.device))
                
                if self.config.get('task_type') == 'classification':
                    pred = output.argmax(dim=1)
                else:
                    pred = output.squeeze()
                
                noisy_predictions.extend(pred.cpu().numpy())
        
        # Compare with original predictions
        original_predictions = []
        with torch.no_grad():
            for data, _ in test_data:
                output = self.model(data.to(self.device))
                
                if self.config.get('task_type') == 'classification':
                    pred = output.argmax(dim=1)
                else:
                    pred = output.squeeze()
                
                original_predictions.extend(pred.cpu().numpy())
        
        # Calculate robustness metrics
        prediction_difference = np.mean(np.abs(np.array(noisy_predictions) - np.array(original_predictions)))
        
        return {
            'noise_level': noise_level,
            'prediction_difference': float(prediction_difference),
            'robustness_score': max(0, 1 - prediction_difference)  # Simple robustness score
        }
    
    def _evaluate_with_missing_features(self, test_data: DataLoader) -> Dict[str, Any]:
        """Evaluate model with randomly missing features."""
        missing_ratios = [0.1, 0.2, 0.5]
        results = {}
        
        for missing_ratio in missing_ratios:
            self.model.eval()
            corrupted_predictions = []
            
            with torch.no_grad():
                for data, _ in test_data:
                    # Randomly set features to zero (simulating missing data)
                    corrupted_data = data.clone()
                    if len(data.shape) > 1:
                        num_features = data.shape[1]
                        num_to_corrupt = int(num_features * missing_ratio)
                        
                        for i in range(data.shape[0]):
                            corrupt_indices = np.random.choice(
                                num_features, num_to_corrupt, replace=False
                            )
                            corrupted_data[i, corrupt_indices] = 0
                    
                    output = self.model(corrupted_data.to(self.device))
                    
                    if self.config.get('task_type') == 'classification':
                        pred = output.argmax(dim=1)
                    else:
                        pred = output.squeeze()
                    
                    corrupted_predictions.extend(pred.cpu().numpy())
            
            results[f'missing_{missing_ratio}'] = {
                'missing_ratio': missing_ratio,
                'num_predictions': len(corrupted_predictions),
                'prediction_stats': {
                    'mean': np.mean(corrupted_predictions),
                    'std': np.std(corrupted_predictions)
                }
            }
        
        return results
    
    def _assess_model_complexity(self) -> Dict[str, Any]:
        """Assess model complexity for interpretability."""
        complexity = {
            'num_parameters': self.model.count_parameters(),
            'num_layers': 0,
            'complexity_level': 'unknown'
        }
        
        # Count layers
        layer_count = 0
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU)):
                layer_count += 1
        
        complexity['num_layers'] = layer_count
        
        # Classify complexity
        if complexity['num_parameters'] < 10000:
            complexity['complexity_level'] = 'low'
        elif complexity['num_parameters'] < 1000000:
            complexity['complexity_level'] = 'medium'
        else:
            complexity['complexity_level'] = 'high'
        
        return complexity
    
    def _calculate_data_quality_score(self,
                                    issues: List[str],
                                    warnings: List[str],
                                    statistics: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score = 1.0
        
        # Penalize for issues and warnings
        score -= len(issues) * 0.2
        score -= len(warnings) * 0.1
        
        # Check data distribution health
        if 'data_stats' in statistics:
            data_stats = statistics['data_stats']
            
            # Check for reasonable variance
            stds = np.array(data_stats['std'])
            if np.any(stds < 1e-6):  # Very low variance
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_data_quality_recommendations(self,
                                             issues: List[str],
                                             warnings: List[str]) -> List[str]:
        """Generate recommendations based on data quality issues."""
        recommendations = []
        
        for issue in issues:
            if "NaN" in issue:
                recommendations.append("Handle missing values through imputation or removal")
            elif "infinite" in issue:
                recommendations.append("Handle infinite values through clipping or transformation")
        
        for warning in warnings:
            if "constant features" in warning:
                recommendations.append("Remove constant features to improve model efficiency")
            elif "class imbalance" in warning:
                recommendations.append("Consider resampling techniques or class weighting")
        
        return recommendations
    
    def _save_validation_results(self, validation_results: Dict[str, Any]) -> None:
        """Save validation results to file."""
        # Save as JSON
        results_path = self.output_dir / f"validation_results_{validation_results['validation_id']}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(validation_results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Validation results saved to {results_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_validation_visualizations(self, validation_results: Dict[str, Any]) -> None:
        """Generate comprehensive validation visualizations."""
        try:
            # Basic performance visualization
            if 'basic_evaluation' in validation_results:
                metrics = validation_results['basic_evaluation']['metrics']
                
                # Create metrics visualization
                fig, ax = plt.subplots(figsize=(12, 8))
                
                metric_names = []
                metric_values = []
                
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and not metric.endswith('_matrix'):
                        metric_names.append(metric.replace('_', ' ').title())
                        metric_values.append(value)
                
                if metric_names:
                    bars = ax.bar(metric_names, metric_values, alpha=0.7)
                    ax.set_title('Model Performance Metrics')
                    ax.set_ylabel('Metric Value')
                    
                    # Color bars based on performance
                    for i, (bar, value) in enumerate(zip(bars, metric_values)):
                        if 'loss' in metric_names[i].lower():
                            color = 'red' if value > 1.0 else 'orange' if value > 0.5 else 'green'
                        else:
                            color = 'green' if value > 0.8 else 'orange' if value > 0.6 else 'red'
                        bar.set_color(color)
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Cross-validation results
            if 'cross_validation' in validation_results and validation_results['cross_validation']['status'] == 'success':
                cv_results = validation_results['cross_validation']['results']
                if 'fold_results' in cv_results:
                    self._plot_cv_results(cv_results['fold_results'])
            
            # Robustness visualization
            if 'robustness' in validation_results:
                self._plot_robustness_results(validation_results['robustness'])
            
            self.logger.info("Validation visualizations generated successfully")
            
        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")
    
    def _plot_cv_results(self, fold_results: List[Dict[str, Any]]) -> None:
        """Plot cross-validation results."""
        # Extract metrics across folds
        metrics_by_fold = {}
        
        for fold_idx, fold_result in enumerate(fold_results):
            for metric, value in fold_result.items():
                if isinstance(value, (int, float)):
                    if metric not in metrics_by_fold:
                        metrics_by_fold[metric] = []
                    metrics_by_fold[metric].append(value)
        
        # Create box plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for metric, values in metrics_by_fold.items():
            if plot_idx < len(axes) and len(values) > 1:
                axes[plot_idx].boxplot(values)
                axes[plot_idx].set_title(f'{metric.replace("_", " ").title()}\nCV Results')
                axes[plot_idx].set_ylabel('Value')
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_validation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_results(self, robustness_results: Dict[str, Any]) -> None:
        """Plot robustness test results."""
        if 'noise_robustness' in robustness_results:
            noise_results = robustness_results['noise_robustness']
            
            noise_levels = []
            robustness_scores = []
            
            for test_name, test_result in noise_results.items():
                if 'robustness_score' in test_result:
                    noise_levels.append(test_result.get('noise_level', 0))
                    robustness_scores.append(test_result['robustness_score'])
            
            if noise_levels and robustness_scores:
                plt.figure(figsize=(10, 6))
                plt.plot(noise_levels, robustness_scores, 'o-', linewidth=2, markersize=8)
                plt.xlabel('Noise Level')
                plt.ylabel('Robustness Score')
                plt.title('Model Robustness to Input Noise')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

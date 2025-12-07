"""
EchoAI Model Registry Pipeline
Complete implementation with validation, bias detection, and GCP push
"""

import os
import sys
import json
import pickle
import joblib
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from google.cloud import aiplatform
from google.cloud import storage
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelRegistryPipeline:
    """Complete Model Registry Pipeline for EchoAI"""
    
    def __init__(self, config_path: str = "Model-Pipeline/config.yaml"):
        """Initialize the pipeline with configuration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # GCP Configuration
        self.project_id = self.config['gcp']['project_id']
        self.location = self.config['gcp']['location']
        self.bucket_name = self.config['gcp']['bucket_name']
        
        # MLflow Configuration
        self.mlflow_tracking_uri = "./mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project_id,
            location=self.location,
            staging_bucket=f"gs://{self.bucket_name}"
        )
        
        # Storage client
        self.storage_client = storage.Client()
        
        print("✅ Model Registry Pipeline Initialized")
        print(f"   Project: {self.project_id}")
        print(f"   MLflow: {self.mlflow_tracking_uri}")
    
    def load_data_from_pipeline(self, data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load processed data from the data pipeline
        Requirement: Loading Data from the Data Pipeline
        """
        print("\n📊 Loading data from data pipeline...")
        
        if data_path is None:
            # Default path from your data pipeline output
            data_path = "Data-Pipeline/processed_data/"
        
        # Load features and labels
        X_val = pd.read_csv(f"{data_path}/X_validation.csv")
        y_val = pd.read_csv(f"{data_path}/y_validation.csv")
        
        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.iloc[:, 0]  # Get first column as Series
        
        print(f"   Loaded {len(X_val)} validation samples")
        print(f"   Features: {X_val.shape[1]}")
        
        return X_val, y_val
    
    def select_best_model(self, experiment_name: Optional[str] = None) -> Dict:
        """
        Select best model from MLflow experiments
        Requirement: Training and Selecting the Best Model
        """
        print("\n🏆 Selecting best model from MLflow...")
        
        # Get all experiments or specific one
        if experiment_name:
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            experiment_ids = [experiment.experiment_id] if experiment else []
        else:
            experiments = self.mlflow_client.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments]
        
        # Search for best run based on F1 score
        best_run = None
        best_metric = -1
        metric_name = self.config.get('model_selection', {}).get('metric', 'f1_score')
        
        for exp_id in experiment_ids:
            runs = self.mlflow_client.search_runs(
                experiment_ids=[exp_id],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=1
            )
            
            if runs and runs[0].data.metrics.get(metric_name, 0) > best_metric:
                best_run = runs[0]
                best_metric = runs[0].data.metrics.get(metric_name, 0)
        
        if not best_run:
            raise ValueError("No suitable model found in MLflow")
        
        print(f"   Best Run ID: {best_run.info.run_id}")
        print(f"   {metric_name}: {best_metric:.4f}")
        print(f"   All metrics: {best_run.data.metrics}")
        
        return {
            'run_id': best_run.info.run_id,
            'metrics': best_run.data.metrics,
            'params': best_run.data.params,
            'experiment_id': best_run.info.experiment_id
        }
    
    def validate_model(self, model_path: str, X_val: pd.DataFrame, 
                       y_val: pd.Series) -> Dict:
        """
        Validate model performance
        Requirement: Model Validation
        """
        print("\n🔍 Validating model performance...")
        
        # Load model
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            # Try MLflow model
            model = mlflow.pyfunc.load_model(model_path)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
        }
        
        # Add AUC if binary classification
        if len(np.unique(y_val)) == 2:
            try:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                metrics['auc'] = roc_auc_score(y_val, y_pred_proba)
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        print("   Validation Metrics:")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"     {key}: {value:.4f}")
        
        # Check thresholds from config
        thresholds = self.config.get('validation_thresholds', {})
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                raise ValueError(f"❌ {metric} ({metrics[metric]:.4f}) below threshold ({threshold})")
        
        print("   ✅ All validation checks passed!")
        return metrics
    
    def detect_bias(self, model_path: str, X_test: pd.DataFrame, 
                    y_test: pd.Series) -> Dict:
        """
        Detect model bias across data slices
        Requirement: Model Bias Detection (Using Slicing Techniques)
        """
        print("\n⚖️ Performing bias detection...")
        
        # Load model
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.endswith('.joblib'):
            model = joblib.load(model_path)
        else:
            model = mlflow.pyfunc.load_model(model_path)
        
        # Get sensitive features from config
        sensitive_features = self.config.get('bias_detection', {}).get('sensitive_features', [])
        
        bias_report = {
            'timestamp': datetime.now().isoformat(),
            'sensitive_features': sensitive_features,
            'bias_detected': False,
            'slices_analyzed': {},
            'recommendations': []
        }
        
        for feature in sensitive_features:
            if feature not in X_test.columns:
                print(f"   ⚠️ Feature '{feature}' not found in data")
                continue
            
            print(f"\n   Analyzing '{feature}'...")
            
            slice_metrics = {}
            unique_values = X_test[feature].dropna().unique()
            
            for value in unique_values:
                # Get slice
                mask = X_test[feature] == value
                X_slice = X_test[mask]
                y_slice = y_test[mask]
                
                if len(y_slice) < 10:  # Skip small slices
                    continue
                
                # Predict and evaluate
                y_pred = model.predict(X_slice)
                
                slice_metrics[str(value)] = {
                    'size': len(y_slice),
                    'accuracy': accuracy_score(y_slice, y_pred),
                    'f1_score': f1_score(y_slice, y_pred, average='weighted', zero_division=0),
                    'precision': precision_score(y_slice, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_slice, y_pred, average='weighted', zero_division=0)
                }
                
                print(f"     {value}: size={len(y_slice)}, "
                      f"acc={slice_metrics[str(value)]['accuracy']:.3f}, "
                      f"f1={slice_metrics[str(value)]['f1_score']:.3f}")
            
            # Calculate disparities
            if len(slice_metrics) > 1:
                accuracies = [m['accuracy'] for m in slice_metrics.values()]
                f1_scores = [m['f1_score'] for m in slice_metrics.values()]
                
                max_diff_accuracy = max(accuracies) - min(accuracies)
                max_diff_f1 = max(f1_scores) - min(f1_scores)
                
                # Check bias threshold
                bias_threshold = self.config.get('bias_detection', {}).get('max_difference', 0.1)
                
                if max_diff_accuracy > bias_threshold or max_diff_f1 > bias_threshold:
                    bias_report['bias_detected'] = True
                    bias_report['recommendations'].append(
                        f"Consider rebalancing data for '{feature}' - "
                        f"accuracy diff: {max_diff_accuracy:.3f}, f1 diff: {max_diff_f1:.3f}"
                    )
                
                bias_report['slices_analyzed'][feature] = {
                    'slices': slice_metrics,
                    'max_accuracy_difference': max_diff_accuracy,
                    'max_f1_difference': max_diff_f1,
                    'bias_detected': max_diff_accuracy > bias_threshold
                }
        
        if bias_report['bias_detected']:
            print("\n   ⚠️ Bias detected in model predictions!")
            for rec in bias_report['recommendations']:
                print(f"   - {rec}")
        else:
            print("\n   ✅ No significant bias detected!")
        
        return bias_report
    
    def push_to_registry(self, model_info: Dict, validation_metrics: Dict, 
                        bias_report: Dict) -> aiplatform.Model:
        """
        Push validated model to Vertex AI Model Registry
        Requirement: Pushing the Model to Artifact or Model Registry
        """
        print("\n🚀 Pushing model to Vertex AI Model Registry...")
        
        run_id = model_info['run_id']
        
        # Download model from MLflow
        print("   Downloading model from MLflow...")
        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path="temp_model"
        )
        
        # Find actual model file
        model_files = list(Path(model_path).rglob("*.pkl")) + \
                     list(Path(model_path).rglob("*.joblib"))
        
        if not model_files:
            # Try to load and save MLflow model
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            model_file = Path("temp_model") / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            model_files = [model_file]
        
        actual_model_path = str(model_files[0])
        
        # Create GCS bucket if needed
        try:
            bucket = self.storage_client.create_bucket(
                self.bucket_name, 
                location=self.location
            )
            print(f"   Created bucket: {self.bucket_name}")
        except:
            bucket = self.storage_client.bucket(self.bucket_name)
            print(f"   Using existing bucket: {self.bucket_name}")
        
        # Generate version
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        model_name = self.config.get('model', {}).get('name', 'echoai-model')
        
        # Upload to GCS
        gcs_folder = f"models/{model_name}/{version}"
        
        # Upload model
        model_blob = bucket.blob(f"{gcs_folder}/model.pkl")
        model_blob.upload_from_filename(actual_model_path)
        print(f"   Uploaded model to GCS")
        
        # Create comprehensive metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'mlflow_run_id': run_id,
            'mlflow_metrics': model_info['metrics'],
            'mlflow_params': model_info['params'],
            'validation_metrics': validation_metrics,
            'bias_report': bias_report,
            'created_at': datetime.now().isoformat(),
            'passed_validation': True,
            'passed_bias_check': not bias_report['bias_detected']
        }
        
        # Upload metadata
        metadata_blob = bucket.blob(f"{gcs_folder}/metadata.json")
        metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
        print(f"   Uploaded metadata to GCS")
        
        # Register in Vertex AI
        print("   Registering in Vertex AI...")
        
        vertex_model = aiplatform.Model.upload(
            display_name=f"{model_name}-{version}",
            artifact_uri=f"gs://{self.bucket_name}/{gcs_folder}",
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
            labels={
                'project': 'echoai',
                'version': version.replace('.', '_'),
                'mlflow_run': run_id[:8],
                'bias_checked': 'true' if not bias_report['bias_detected'] else 'false',
                'f1_score': str(int(validation_metrics['f1_score'] * 100))
            },
            description=f"""
            EchoAI Model - {model_name}
            Version: {version}
            MLflow Run: {run_id}
            F1 Score: {validation_metrics['f1_score']:.4f}
            Accuracy: {validation_metrics['accuracy']:.4f}
            Bias Check: {'Passed ✅' if not bias_report['bias_detected'] else 'Warning ⚠️'}
            Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        )
        
        print(f"\n   ✅ Model successfully registered!")
        print(f"   Display Name: {vertex_model.display_name}")
        print(f"   Resource Name: {vertex_model.resource_name}")
        print(f"   Console URL: https://console.cloud.google.com/vertex-ai/models")
        
        # Clean up temp files
        import shutil
        if os.path.exists("temp_model"):
            shutil.rmtree("temp_model")
        
        return vertex_model
    
    def run_complete_pipeline(self, data_path: Optional[str] = None):
        """
        Run the complete model registry pipeline
        """
        print("\n" + "="*70)
        print("🚀 ECHOAI MODEL REGISTRY PIPELINE")
        print("="*70)
        
        try:
            # Step 1: Load data from pipeline
            X_val, y_val = self.load_data_from_pipeline(data_path)
            
            # Step 2: Select best model from MLflow
            model_info = self.select_best_model()
            
            # Step 3: Download and prepare model path
            model_path = f"runs:/{model_info['run_id']}/model"
            
            # Step 4: Validate model
            validation_metrics = self.validate_model(model_path, X_val, y_val)
            
            # Step 5: Check for bias
            bias_report = self.detect_bias(model_path, X_val, y_val)
            
            # Step 6: Push to registry if all checks pass
            if validation_metrics and not self.config.get('bias_detection', {}).get('block_on_bias', False):
                vertex_model = self.push_to_registry(
                    model_info, 
                    validation_metrics, 
                    bias_report
                )
                
                print("\n" + "="*70)
                print("🎉 SUCCESS! Model registered in Vertex AI")
                print("="*70)
                
                return {
                    'success': True,
                    'model': vertex_model,
                    'validation_metrics': validation_metrics,
                    'bias_report': bias_report
                }
            else:
                print("\n❌ Model did not pass all checks")
                return {
                    'success': False,
                    'validation_metrics': validation_metrics,
                    'bias_report': bias_report
                }
                
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            return {'success': False, 'error': str(e)}
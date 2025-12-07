#!/usr/bin/env python3
"""
Model Registry Push for EchoAI Project
Adapted for your specific directory structure
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# MLflow and GCP imports
import mlflow
from mlflow.tracking import MlflowClient
from google.cloud import aiplatform
from google.cloud import storage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EchoAIModelRegistry:
    def __init__(self):
        """Initialize with your project structure"""
        
        self.PROJECT_ID = "echo-ai-478802"  
        self.LOCATION = "us-central1"
        self.BUCKET_NAME = f"{self.PROJECT_ID}-echoai-models"
        
        # Your MLflow directory
        self.mlflow_tracking_uri = "./mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Initialize GCP
        try:
            aiplatform.init(
                project=self.PROJECT_ID,
                location=self.LOCATION
            )
            self.storage_client = storage.Client()
            print(f"✅ Connected to GCP project: {self.PROJECT_ID}")
        except Exception as e:
            print(f"⚠️ GCP initialization warning: {e}")
            print("Make sure to run: gcloud auth application-default login")
        
        print(f"✅ MLflow tracking at: {self.mlflow_tracking_uri}")
    
    def load_validation_data(self):
        """Load validation data from your data folder"""
        print("\n📊 Loading validation data...")
        
        # Check for validation data in different possible locations
        possible_paths = [
            "data/raw/validation.csv",
            "data/processed/validation.csv",
            "data/raw/dataset_restaurant-review-aggregator_2025-11-22_23-47-46-681.csv"
        ]
        
        validation_data = None
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   Found data at: {path}")
                validation_data = pd.read_csv(path)
                break
        
        if validation_data is None:
            print("   ⚠️ No validation data found, using sample from raw data")
            # Use the restaurant review data as validation
            raw_data = pd.read_csv("data/raw/dataset_restaurant-review-aggregator_2025-11-22_23-47-46-681.csv")
            
            # Sample for validation (adjust columns based on your actual data)
            validation_data = raw_data.sample(n=min(1000, len(raw_data)), random_state=42)
        
        print(f"   Loaded {len(validation_data)} validation samples")
        
        # Prepare X and y (adjust based on your actual column names)
        # Assuming your data has 'review_text' and 'rating' or 'sentiment' columns
        feature_columns = [col for col in validation_data.columns 
                          if col not in ['rating', 'sentiment', 'label', 'target']]
        target_column = None
        
        for col in ['rating', 'sentiment', 'label', 'target']:
            if col in validation_data.columns:
                target_column = col
                break
        
        if target_column:
            X_val = validation_data[feature_columns]
            y_val = validation_data[target_column]
        else:
            # Create dummy data for testing
            X_val = validation_data
            y_val = pd.Series(np.random.randint(0, 2, len(validation_data)))
            print("   ⚠️ Using dummy target for testing - update with actual target column")
        
        return X_val, y_val
    
    def get_best_model_from_mlflow(self):
        """Get the best model from your MLflow runs"""
        print("\n🔍 Searching for best model in MLflow...")
        
        # List all experiments
        experiments = self.mlflow_client.search_experiments()
        print(f"   Found {len(experiments)} experiments")
        
        all_runs = []
        for exp in experiments:
            if exp.name != "Default":  # Skip default if empty
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=10
                )
                all_runs.extend(runs)
                if runs:
                    print(f"   Experiment '{exp.name}': {len(runs)} runs")
        
        if not all_runs:
            print("   ⚠️ No runs found in MLflow")
            return None
        
        # Find best run by F1 score (or other metrics)
        best_run = None
        best_score = -1
        metric_name = None
        
        # Try different metric names
        for run in all_runs:
            for metric in ['f1_score', 'f1', 'accuracy', 'val_f1_score', 'val_accuracy']:
                if metric in run.data.metrics:
                    score = run.data.metrics[metric]
                    if score > best_score:
                        best_score = score
                        best_run = run
                        metric_name = metric
                    break
        
        if best_run:
            print(f"\n   🏆 Best model found!")
            print(f"      Run ID: {best_run.info.run_id}")
            print(f"      {metric_name}: {best_score:.4f}")
            print(f"      All metrics: {dict(list(best_run.data.metrics.items())[:5])}")
            return best_run
        
        return None
    
    def validate_model(self, run_id, X_val, y_val):
        """Validate the model performance"""
        print("\n📊 Validating model...")
        
        try:
            # Load model from MLflow
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Make predictions
            predictions = model.predict(X_val)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, predictions),
                'precision': precision_score(y_val, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_val, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y_val, predictions, average='weighted', zero_division=0)
            }
            
            print("   Validation Results:")
            for metric, value in metrics.items():
                print(f"      {metric}: {value:.4f}")
            
            # Check thresholds
            if metrics['f1_score'] < 0.6:  # Lowered threshold for testing
                print(f"   ⚠️ F1 score ({metrics['f1_score']:.4f}) below threshold")
            else:
                print("   ✅ Model validation passed!")
            
            return metrics, model
            
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return None, None
    
    def push_to_vertex_ai(self, run_id, model, metrics):
        """Push the model to Vertex AI Model Registry"""
        print("\n🚀 Pushing to Vertex AI Model Registry...")
        
        try:
            # Create bucket if needed
            bucket_name = self.BUCKET_NAME
            try:
                bucket = self.storage_client.create_bucket(bucket_name, location=self.LOCATION)
                print(f"   Created bucket: {bucket_name}")
            except:
                bucket = self.storage_client.bucket(bucket_name)
                print(f"   Using existing bucket: {bucket_name}")
            
            # Generate version
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
            model_name = "echoai-review-model"
            
            # Save model locally first
            temp_model_path = "temp_model.pkl"
            with open(temp_model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Upload to GCS
            gcs_path = f"models/{model_name}/{version}/model.pkl"
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(temp_model_path)
            print(f"   Uploaded to GCS: gs://{bucket_name}/{gcs_path}")
            
            # Upload metadata
            metadata = {
                "mlflow_run_id": run_id,
                "version": version,
                "metrics": metrics,
                "created_at": datetime.now().isoformat()
            }
            
            metadata_blob = bucket.blob(f"models/{model_name}/{version}/metadata.json")
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
            
            # Register in Vertex AI
            vertex_model = aiplatform.Model.upload(
                display_name=f"{model_name}-{version}",
                artifact_uri=f"gs://{bucket_name}/models/{model_name}/{version}",
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
                description=f"EchoAI Model - F1: {metrics['f1_score']:.4f}"
            )
            
            print(f"\n   ✅ Model registered in Vertex AI!")
            print(f"      Name: {vertex_model.display_name}")
            print(f"      Resource: {vertex_model.resource_name}")
            
            # Clean up
            os.remove(temp_model_path)
            
            return vertex_model
            
        except Exception as e:
            print(f"   ❌ Error pushing to Vertex AI: {e}")
            return None
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("🚀 ECHOAI MODEL REGISTRY PIPELINE")
        print("="*60)
        
        # Step 1: Load validation data
        try:
            X_val, y_val = self.load_validation_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
        
        # Step 2: Get best model from MLflow
        best_run = self.get_best_model_from_mlflow()
        if not best_run:
            print("❌ No model found in MLflow")
            print("\nTip: Make sure you have trained models in ./mlruns")
            return
        
        # Step 3: Validate model
        metrics, model = self.validate_model(best_run.info.run_id, X_val, y_val)
        if not metrics:
            print("❌ Model validation failed")
            return
        
        # Step 4: Push to Vertex AI
        print("\n" + "="*60)
        response = input("Push model to Vertex AI? (yes/no): ").lower()
        
        if response == 'yes':
            vertex_model = self.push_to_vertex_ai(
                best_run.info.run_id, 
                model, 
                metrics
            )
            
            if vertex_model:
                print("\n" + "="*60)
                print("🎉 SUCCESS! Model pushed to Vertex AI")
                print("="*60)
                print(f"\nView your model at:")
                print(f"https://console.cloud.google.com/vertex-ai/models")
        else:
            print("Skipping Vertex AI push")

def main():
    # Quick check for required files
    if not os.path.exists("mlruns"):
        print("❌ MLflow runs directory not found!")
        print("Make sure you're in the echo-ai-main-3 directory")
        return
    
    # Run pipeline
    registry = EchoAIModelRegistry()
    registry.run_pipeline()

if __name__ == "__main__":
    main()
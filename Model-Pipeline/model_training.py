"""
Model training module for EchoAI
Trains multiple models and selects the best one
"""
import numpy as np
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime
from config import *
from data_loader import prepare_data_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            min_df=2,
            max_df=0.95
        )
        self.best_model = None
        self.best_score = -1
        self.models_to_train = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE
            ),
            'LinearSVC': LinearSVC(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ),
            'MultinomialNB': MultinomialNB(alpha=0.1)
        }
    
    def prepare_features(self, X_train, X_val, X_test):
        """Convert text to TF-IDF features"""
        logger.info("Vectorizing text data...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_val_tfidf = self.vectorizer.transform(X_val)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        logger.info(f"Feature matrix shape: {X_train_tfidf.shape}")
        return X_train_tfidf, X_val_tfidf, X_test_tfidf
    
    def evaluate_model(self, model, X, y, dataset_name=""):
        """Evaluate model performance"""
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        metrics = {
            f'{dataset_name}_accuracy': accuracy,
            f'{dataset_name}_precision': precision,
            f'{dataset_name}_recall': recall,
            f'{dataset_name}_f1': f1
        }
        
        return metrics, y_pred
    
    def train_single_model(self, model_name, model, X_train, y_train, X_val, y_val):
        """Train a single model and track with MLflow"""
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info(f"Training {model_name}...")
            
            # Log model type
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("max_features", MAX_FEATURES)
            mlflow.log_param("ngram_range", str(NGRAM_RANGE))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            train_metrics, _ = self.evaluate_model(model, X_train, y_train, "train")
            for key, value in train_metrics.items():
                mlflow.log_metric(key, value)
            
            # Evaluate on validation set
            val_metrics, val_pred = self.evaluate_model(model, X_val, y_val, "val")
            for key, value in val_metrics.items():
                mlflow.log_metric(key, value)
            
            # Log confusion matrix
            cm = confusion_matrix(y_val, val_pred)
            mlflow.log_text(str(cm), "confusion_matrix.txt")
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            logger.info(f"{model_name} - Val F1: {val_metrics['val_f1']:.4f}")
            
            return val_metrics['val_f1'], model
    
    def train_all_models(self, data_splits):
        """Train all models and select the best one"""
        # Prepare data
        X_train, y_train, _ = data_splits['train']
        X_val, y_val, _ = data_splits['val']
        X_test, y_test, _ = data_splits['test']

        # Vectorize
        X_train_tfidf, X_val_tfidf, X_test_tfidf = self.prepare_features(
            X_train, X_val, X_test
        )
        
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        results = {}
        
        # Train each model
        for model_name, model in self.models_to_train.items():
            try:
                f1_score, trained_model = self.train_single_model(
                    model_name, model,
                    X_train_tfidf, y_train,
                    X_val_tfidf, y_val
                )
                
                results[model_name] = {
                    'f1_score': f1_score,
                    'model': trained_model
                }
                
                # Track best model
                if f1_score > self.best_score:
                    self.best_score = f1_score
                    self.best_model = trained_model
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Evaluate best model on test set
        if self.best_model:
            logger.info(f"\nBest model: {self.best_model_name} with F1: {self.best_score:.4f}")
            test_metrics, test_pred = self.evaluate_model(
                self.best_model, X_test_tfidf, y_test, "test"
            )
            logger.info(f"Test set performance: {test_metrics}")
            
            # Save best model
            joblib.dump(self.best_model, BEST_MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)
            logger.info(f"Best model saved to {BEST_MODEL_PATH}")
            
            return self.best_model, self.vectorizer, results
        
        return None, None, results

def main():
    """Main training pipeline"""
    
    logger.info("="*60)
    logger.info("Starting EchoAI Model Training Pipeline")
    logger.info("="*60)
    
    # Load and prepare data
    data_splits, stats = prepare_data_for_training()
    
    # Initialize trainer
    trainer = SentimentModelTrainer()
    
    # Train models
    best_model, vectorizer, results = trainer.train_all_models(data_splits)
    
    # Save results summary
    import json
    results_summary = {
        'best_model': trainer.best_model_name if hasattr(trainer, 'best_model_name') else None,
        'best_f1_score': float(trainer.best_score),
        'model_scores': {k: float(v['f1_score']) for k, v in results.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(RESULTS_DIR / 'training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("="*60)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info("="*60)
    
    return best_model, vectorizer, results

if __name__ == "__main__":
    main()
"""
Hyperparameter tuning module using Optuna
"""
import optuna
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import mlflow
import joblib
import logging
from config import *
from data_loader import prepare_data_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, model_type='LogisticRegression'):
        self.model_type = model_type
        self.best_params = None
        self.best_score = -1
        self.study = None
        
    def get_search_space(self, trial):
        """Define hyperparameter search space for each model type"""
        if self.model_type == 'LogisticRegression':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
        
        elif self.model_type == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
        
        elif self.model_type == 'GradientBoosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
    
    def create_model(self, params):
        """Create model instance with given parameters"""
        if self.model_type == 'LogisticRegression':
            return LogisticRegression(random_state=RANDOM_STATE, **params)
        elif self.model_type == 'RandomForest':
            return RandomForestClassifier(random_state=RANDOM_STATE, **params)
        elif self.model_type == 'GradientBoosting':
            return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    
    def objective(self, trial, X_train, y_train):
        """Objective function for Optuna optimization"""
        # Get hyperparameters
        params = self.get_search_space(trial)
        
        # Create model
        model = self.create_model(params)
        
        # Use cross-validation for robust evaluation
        f1_scorer = make_scorer(f1_score, average='weighted')
        scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS,
            scoring=f1_scorer,
            n_jobs=-1
        )
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1_mean", scores.mean())
            mlflow.log_metric("cv_f1_std", scores.std())
        
        return scores.mean()
    
    def tune_hyperparameters(self, X_train, y_train, n_trials=None):
        """Run hyperparameter optimization"""
        n_trials = n_trials or HYPERPARAMETER_SEARCH_TRIALS
        
        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        logger.info(f"Running {n_trials} trials with {CV_FOLDS}-fold CV")
        
        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_NAME}_tuning")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.model_type}_optimization'
        )
        
        # Run optimization
        with mlflow.start_run(run_name=f"{self.model_type}_tuning"):
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", CV_FOLDS)
            
            self.study.optimize(
                lambda trial: self.objective(trial, X_train, y_train),
                n_trials=n_trials
            )
            
            # Get best parameters
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            # Log best results
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_cv_f1", self.best_score)
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV F1 score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def train_best_model(self, X_train, y_train, X_val, y_val):
        """Train model with best hyperparameters"""
        if self.best_params is None:
            raise ValueError("No best parameters found. Run tune_hyperparameters first.")
        
        logger.info("Training model with best parameters...")
        best_model = self.create_model(self.best_params)
        best_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        from sklearn.metrics import accuracy_score, f1_score
        val_pred = best_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        logger.info(f"Validation F1: {val_f1:.4f}")
        
        return best_model
    
    def save_tuning_results(self):
        """Save tuning results and visualization"""
        import json
        
        results = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'n_trials': len(self.study.trials)
        }
        
        with open(RESULTS_DIR / f'{self.model_type}_tuning_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create optimization history plot
        try:
            import matplotlib.pyplot as plt
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_html(str(RESULTS_DIR / f'{self.model_type}_optimization_history.html'))
            
            # Parameter importance
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_html(str(RESULTS_DIR / f'{self.model_type}_param_importance.html'))
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")

def tune_best_model(model_type='LogisticRegression'):
    """Main function to tune hyperparameters for a specific model"""
    logger.info("="*60)
    logger.info(f"Hyperparameter Tuning for {model_type}")
    logger.info("="*60)
    
    # Load data
    data_splits, _ = prepare_data_for_training()
    X_train, y_train, _ = data_splits['train']
    X_val, y_val, _ = data_splits['val']
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Initialize tuner
    tuner = HyperparameterTuner(model_type)
    
    # Run tuning
    best_params, best_score = tuner.tune_hyperparameters(X_train_tfidf, y_train)
    
    # Train best model
    best_model = tuner.train_best_model(X_train_tfidf, y_train, X_val_tfidf, y_val)
    
    # Save results
    tuner.save_tuning_results()
    
    # Save tuned model
    model_path = MODEL_DIR / f'{model_type}_tuned.pkl'
    joblib.dump(best_model, model_path)
    logger.info(f"Tuned model saved to {model_path}")
    
    return best_model, best_params

if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'LogisticRegression'
    tune_best_model(model_type)
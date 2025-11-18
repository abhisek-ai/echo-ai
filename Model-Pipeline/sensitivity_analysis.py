"""
Sensitivity analysis and model interpretability using SHAP and LIME
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
from config import *
from data_loader import prepare_data_for_training

# Import interpretability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not installed. Install with: pip install lime")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    def __init__(self, model=None, vectorizer=None):
        if model is None:
            self.model = joblib.load(BEST_MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            self.model = model
            self.vectorizer = vectorizer
        
        self.class_names = ['negative', 'neutral', 'positive']
        self.feature_names = None
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            self.feature_names = self.vectorizer.get_feature_names_out()
        
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based and linear models"""
        importance_data = {}
        
        # Check model type and extract importance
        if hasattr(self.model, 'coef_'):
            # Linear model (LogisticRegression, LinearSVC)
            logger.info("Analyzing feature importance for linear model")
            
            if len(self.model.coef_.shape) > 1:
                # Multi-class: aggregate importance across classes
                importance = np.abs(self.model.coef_).mean(axis=0)
                
                # Also get per-class importance
                for i, class_name in enumerate(self.class_names):
                    class_importance = self.model.coef_[i]
                    top_positive = self.get_top_features(class_importance, n=20, positive=True)
                    top_negative = self.get_top_features(class_importance, n=20, positive=False)
                    importance_data[f'{class_name}_positive'] = top_positive
                    importance_data[f'{class_name}_negative'] = top_negative
            else:
                importance = np.abs(self.model.coef_[0])
            
            # Get overall top features
            top_features = self.get_top_features(importance, n=30)
            importance_data['overall_top_features'] = top_features
            
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based model (RandomForest, GradientBoosting)
            logger.info("Analyzing feature importance for tree-based model")
            importance = self.model.feature_importances_
            top_features = self.get_top_features(importance, n=30)
            importance_data['overall_top_features'] = top_features
        else:
            logger.warning("Model does not have feature importance attributes")
            return None
        
        # Create visualization
        self.plot_feature_importance(importance_data)
        
        return importance_data
    
    def get_top_features(self, importance_scores, n=20, positive=True):
        """Get top n important features"""
        if self.feature_names is None:
            return []
        
        if positive:
            indices = np.argsort(importance_scores)[-n:][::-1]
        else:
            indices = np.argsort(importance_scores)[:n]
        
        top_features = []
        for idx in indices:
            if idx < len(self.feature_names):
                top_features.append({
                    'feature': self.feature_names[idx],
                    'importance': float(importance_scores[idx])
                })
        
        return top_features
    
    def plot_feature_importance(self, importance_data, save_path=None):
        """Plot feature importance visualization"""
        if 'overall_top_features' not in importance_data:
            return
        
        top_features = importance_data['overall_top_features'][:20]
        features = [f['feature'] for f in top_features]
        scores = [f['importance'] for f in top_features]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title('Top 20 Most Important Features')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def shap_analysis(self, X_sample, n_samples=100):
        """Perform SHAP analysis for global interpretability"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP analysis.")
            return None
        
        logger.info("Running SHAP analysis...")
        
        # Sample data for efficiency
        if len(X_sample) > n_samples:
            indices = np.random.choice(len(X_sample), n_samples, replace=False)
            X_sample = X_sample[indices]
        
        # Transform text to features
        X_transformed = self.vectorizer.transform(X_sample)
        
        # Create SHAP explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            # Use KernelExplainer for any model with predict_proba
            explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x),
                X_transformed,
                link='logit'
            )
        else:
            # Use linear explainer for linear models
            explainer = shap.LinearExplainer(
                self.model,
                X_transformed,
                feature_dependence='independent'
            )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_transformed)
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            # Multi-class: show for positive class
            shap.summary_plot(
                shap_values[2],  # Positive class
                X_transformed.toarray(),
                feature_names=self.feature_names,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                X_transformed.toarray(),
                feature_names=self.feature_names,
                show=False
            )
        
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values
    
    def lime_explanation(self, text_sample, n_features=10):
        """Generate LIME explanation for a single text sample"""
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Skipping LIME analysis.")
            return None
        
        logger.info("Generating LIME explanation...")
        
        # Create LIME explainer
        explainer = LimeTextExplainer(
            class_names=self.class_names,
            random_state=RANDOM_STATE
        )
        
        # Create prediction function
        def predict_fn(texts):
            X = self.vectorizer.transform(texts)
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # For models without predict_proba, use decision function
                decision = self.model.decision_function(X)
                # Convert to probability-like scores
                proba = np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)
                return proba
        
        # Generate explanation
        explanation = explainer.explain_instance(
            text_sample,
            predict_fn,
            num_features=n_features,
            num_samples=5000
        )
        
        # Get explanation as list
        exp_list = explanation.as_list()
        
        # Create visualization
        fig = explanation.as_pyplot_figure()
        fig.suptitle('LIME Explanation for Sample Text', fontsize=14)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'lime_explanation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return exp_list
    
    def hyperparameter_sensitivity(self, X_train, y_train, X_val, y_val, param_name, param_range):
        """Analyze sensitivity to hyperparameter changes"""
        logger.info(f"Analyzing sensitivity to {param_name}")
        
        from sklearn.metrics import f1_score
        
        results = []
        base_params = self.model.get_params()
        
        for param_value in param_range:
            # Create new model with modified parameter
            model_class = type(self.model)
            new_params = base_params.copy()
            new_params[param_name] = param_value
            
            try:
                new_model = model_class(**new_params)
                
                # Train and evaluate
                X_train_tfidf = self.vectorizer.transform(X_train)
                X_val_tfidf = self.vectorizer.transform(X_val)
                
                new_model.fit(X_train_tfidf, y_train)
                val_pred = new_model.predict(X_val_tfidf)
                
                score = f1_score(y_val, val_pred, average='weighted')
                results.append({
                    'param_value': param_value,
                    'f1_score': score
                })
                
            except Exception as e:
                logger.warning(f"Could not test {param_name}={param_value}: {e}")
                continue
        
        # Plot sensitivity
        if results:
            df_results = pd.DataFrame(results)
            plt.figure(figsize=(10, 6))
            plt.plot(df_results['param_value'], df_results['f1_score'], 'o-', linewidth=2)
            plt.xlabel(param_name)
            plt.ylabel('F1 Score')
            plt.title(f'Model Sensitivity to {param_name}')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f'sensitivity_{param_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return results
    
    def analyze_prediction_confidence(self, X_test, y_test):
        """Analyze model confidence in predictions"""
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return None
        
        logger.info("Analyzing prediction confidence...")
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        y_proba = self.model.predict_proba(X_test_tfidf)
        
        # Get maximum probability (confidence) for each prediction
        max_proba = np.max(y_proba, axis=1)
        
        # Analyze by correctness
        correct = y_pred == y_test
        
        confidence_analysis = {
            'overall_mean_confidence': float(np.mean(max_proba)),
            'correct_mean_confidence': float(np.mean(max_proba[correct])),
            'incorrect_mean_confidence': float(np.mean(max_proba[~correct])) if np.any(~correct) else None,
            'low_confidence_threshold': 0.5,
            'low_confidence_samples': int(np.sum(max_proba < 0.5)),
            'high_confidence_threshold': 0.9,
            'high_confidence_samples': int(np.sum(max_proba > 0.9))
        }
        
        # Plot confidence distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist([max_proba[correct], max_proba[~correct]], 
                bins=30, alpha=0.7, label=['Correct', 'Incorrect'])
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Prediction Correctness')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i
            plt.hist(max_proba[class_mask], bins=20, alpha=0.5, label=class_name)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by True Class')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return confidence_analysis
    
    def run_complete_sensitivity_analysis(self):
        """Run all sensitivity analyses"""
        logger.info("="*60)
        logger.info("Running Complete Sensitivity Analysis")
        logger.info("="*60)
        
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'analyses_performed': []
        }
        
        # Load test data
        data_splits, _ = prepare_data_for_training()
        X_train, y_train, _ = data_splits['train']
        X_val, y_val, _ = data_splits['val']
        X_test, y_test, _ = data_splits['test']
        
        # 1. Feature Importance Analysis
        logger.info("\n1. Analyzing Feature Importance...")
        feature_importance = self.analyze_feature_importance()
        if feature_importance:
            results['feature_importance'] = feature_importance
            results['analyses_performed'].append('feature_importance')
        
        # 2. SHAP Analysis
        if SHAP_AVAILABLE:
            logger.info("\n2. Running SHAP Analysis...")
            shap_values = self.shap_analysis(X_test[:100])
            if shap_values is not None:
                results['analyses_performed'].append('shap')
        
        # 3. LIME Analysis (single example)
        if LIME_AVAILABLE:
            logger.info("\n3. Running LIME Analysis...")
            sample_text = X_test[0]
            lime_exp = self.lime_explanation(sample_text)
            if lime_exp:
                results['lime_example'] = {
                    'text': sample_text[:200],
                    'explanation': lime_exp
                }
                results['analyses_performed'].append('lime')
        
        # 4. Confidence Analysis
        logger.info("\n4. Analyzing Prediction Confidence...")
        confidence_analysis = self.analyze_prediction_confidence(X_test, y_test)
        if confidence_analysis:
            results['confidence_analysis'] = confidence_analysis
            results['analyses_performed'].append('confidence')
        
        # 5. Hyperparameter Sensitivity (example for LogisticRegression)
        if hasattr(self.model, 'C'):  # LogisticRegression
            logger.info("\n5. Analyzing Hyperparameter Sensitivity...")
            C_range = [0.001, 0.01, 0.1, 1, 10, 100]
            sensitivity_results = self.hyperparameter_sensitivity(
                X_train[:1000], y_train[:1000],
                X_val[:500], y_val[:500],
                'C', C_range
            )
            if sensitivity_results:
                results['hyperparameter_sensitivity'] = {
                    'parameter': 'C',
                    'results': sensitivity_results
                }
                results['analyses_performed'].append('hyperparameter_sensitivity')
        
        # Save results
        import json
        with open(RESULTS_DIR / 'sensitivity_analysis_results.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            json.dump(convert_types(results), f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("Sensitivity Analysis Complete!")
        logger.info(f"Analyses performed: {', '.join(results['analyses_performed'])}")
        logger.info(f"Results saved to {RESULTS_DIR}")
        logger.info("="*60)
        
        return results

def main():
    """Main sensitivity analysis pipeline"""
    analyzer = SensitivityAnalyzer()
    results = analyzer.run_complete_sensitivity_analysis()
    return results

if __name__ == "__main__":
    main()
"""
Model validation module for comprehensive evaluation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from config import *
from data_loader import prepare_data_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, model=None, vectorizer=None):
        if model is None:
            self.model = joblib.load(BEST_MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            self.model = model
            self.vectorizer = vectorizer
        
        self.class_names = ['negative', 'neutral', 'positive']
        
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i]
            metrics[f'{class_name}_recall'] = recall[i]
            metrics[f'{class_name}_f1'] = f1[i]
            metrics[f'{class_name}_support'] = int(support[i])
        
        # Weighted average metrics
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        # Macro average metrics
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # AUC if probabilities available
        if y_proba is not None:
            try:
                # Binarize labels for multi-class ROC
                y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
                
                # Calculate AUC for each class
                for i, class_name in enumerate(self.class_names):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    metrics[f'{class_name}_auc'] = auc(fpr, tpr)
                
                # Macro average AUC
                metrics['macro_auc'] = np.mean([metrics[f'{cn}_auc'] for cn in self.class_names])
                
                # Weighted average AUC
                weights = support / support.sum()
                metrics['weighted_auc'] = np.sum([metrics[f'{cn}_auc'] * w 
                                                 for cn, w in zip(self.class_names, weights)])
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        return metrics
    
    def create_confusion_matrix_plot(self, y_true, y_pred, save_path=None):
        """Create and save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.show()
        
        return cm
    
    def create_roc_curves(self, y_true, y_proba, save_path=None):
        """Create ROC curves for multi-class classification"""
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class Sentiment Analysis')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        plt.show()
    
    def analyze_errors(self, X, y_true, y_pred, n_samples=10):
        """Analyze misclassified samples"""
        misclassified = np.where(y_true != y_pred)[0]
        
        if len(misclassified) == 0:
            logger.info("No misclassifications found!")
            return []
        
        logger.info(f"Found {len(misclassified)} misclassifications out of {len(y_true)} samples")
        
        # Sample some errors for analysis
        error_samples = []
        sample_indices = np.random.choice(misclassified, 
                                        min(n_samples, len(misclassified)), 
                                        replace=False)
        
        for idx in sample_indices:
            error_samples.append({
                'text': X[idx][:100] + '...' if len(X[idx]) > 100 else X[idx],
                'true_label': self.class_names[y_true[idx]],
                'predicted_label': self.class_names[y_pred[idx]]
            })
        
        return error_samples
    
    def validate_model(self, X_test, y_test, metadata=None):
        """Comprehensive model validation"""
        logger.info("Starting model validation...")
        
        # Transform text data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Get predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Get probabilities if available
        try:
            y_proba = self.model.predict_proba(X_test_tfidf)
        except:
            y_proba = None
            logger.warning("Model does not support predict_proba")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Create visualizations
        cm = self.create_confusion_matrix_plot(
            y_test, y_pred,
            save_path=RESULTS_DIR / 'confusion_matrix.png'
        )
        
        if y_proba is not None:
            self.create_roc_curves(
                y_test, y_proba,
                save_path=RESULTS_DIR / 'roc_curves.png'
            )
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Analyze errors
        error_samples = self.analyze_errors(X_test, y_test, y_pred)
        
        # Save results
        validation_results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'error_samples': error_samples
        }
        
        import json
        with open(RESULTS_DIR / 'validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info("Validation complete!")
        self.print_results_summary(metrics, report)
        
        return validation_results
    
    def print_results_summary(self, metrics, report):
        """Print formatted results summary"""
        print("\n" + "="*60)
        print("MODEL VALIDATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
        print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
        
        if 'weighted_auc' in metrics:
            print(f"Weighted AUC: {metrics['weighted_auc']:.4f}")
        
        print("\nPer-Class Performance:")
        print("-"*40)
        for class_name in self.class_names:
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {metrics[f'{class_name}_precision']:.4f}")
            print(f"  Recall: {metrics[f'{class_name}_recall']:.4f}")
            print(f"  F1-Score: {metrics[f'{class_name}_f1']:.4f}")
            if f'{class_name}_auc' in metrics:
                print(f"  AUC: {metrics[f'{class_name}_auc']:.4f}")
            print(f"  Support: {metrics[f'{class_name}_support']}")
        
        print("\n" + "="*60)

def main():
    """Main validation pipeline"""
    logger.info("Loading test data...")
    data_splits, _ = prepare_data_for_training()
    X_test, y_test, metadata = data_splits['test']
    
    # Initialize validator
    validator = ModelValidator()
    
    # Run validation
    results = validator.validate_model(X_test, y_test, metadata)
    
    logger.info(f"Results saved to {RESULTS_DIR}")
    
    return results

if __name__ == "__main__":
    main()
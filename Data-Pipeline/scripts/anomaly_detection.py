# Data-Pipeline/scripts/anomaly_detection.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> List[int]:
    """Detect outliers using IQR method"""
    if column not in df.columns:
        return []
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers.index.tolist()

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> List[int]:
    """Detect outliers using Z-score method"""
    if column not in df.columns:
        return []
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df.iloc[np.where(z_scores > threshold)[0]]
    return outliers.index.tolist()

def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect duplicate records"""
    duplicates = {
        'exact_duplicates': df[df.duplicated(keep=False)],
        'duplicate_texts': df[df.duplicated(subset=['text'], keep=False)],
        'duplicate_ids': df[df.duplicated(subset=['review_id'], keep=False)]
    }
    
    duplicate_summary = {
        'exact_duplicate_count': len(duplicates['exact_duplicates']),
        'duplicate_text_count': len(duplicates['duplicate_texts']),
        'duplicate_id_count': len(duplicates['duplicate_ids']),
        'duplicate_indices': {
            'exact': duplicates['exact_duplicates'].index.tolist()[:10],  # First 10
            'text': duplicates['duplicate_texts'].index.tolist()[:10],
            'ids': duplicates['duplicate_ids'].index.tolist()[:10]
        }
    }
    
    return duplicate_summary

def detect_suspicious_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect suspicious patterns in reviews"""
    suspicious_patterns = []
    
    # Check for very short reviews
    if 'text_length' in df.columns:
        short_reviews = df[df['text_length'] < 10]
        if len(short_reviews) > 0:
            suspicious_patterns.append({
                'type': 'very_short_reviews',
                'count': len(short_reviews),
                'indices': short_reviews.index.tolist()[:10],
                'description': 'Reviews with less than 10 characters'
            })
    
    # Check for reviews with extreme votes but low ratings
    if 'useful_votes' in df.columns and 'rating' in df.columns:
        suspicious_votes = df[(df['useful_votes'] > df['useful_votes'].quantile(0.95)) & 
                             (df['rating'] <= 2)]
        if len(suspicious_votes) > 0:
            suspicious_patterns.append({
                'type': 'suspicious_vote_pattern',
                'count': len(suspicious_votes),
                'indices': suspicious_votes.index.tolist()[:10],
                'description': 'Low ratings with unusually high useful votes'
            })
    
    # Check for identical reviews from same business
    if 'business_id' in df.columns and 'text' in df.columns:
        grouped = df.groupby(['business_id', 'text']).size()
        repeated = grouped[grouped > 1]
        if len(repeated) > 0:
            suspicious_patterns.append({
                'type': 'repeated_reviews_per_business',
                'count': len(repeated),
                'description': 'Identical reviews for the same business'
            })
    
    return suspicious_patterns

def detect_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect patterns in missing data"""
    missing_analysis = {}
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(df)) * 100,
                'missing_indices': df[df[col].isnull()].index.tolist()[:10]
            }
    
    return missing_analysis

def generate_anomaly_alert(anomalies: Dict[str, Any]) -> str:
    """Generate alert message for detected anomalies"""
    alert_messages = []
    
    # Check outliers
    total_outliers = sum(len(v) for k, v in anomalies.items() if 'outliers' in k)
    if total_outliers > 0:
        alert_messages.append(f"⚠️ ALERT: {total_outliers} outliers detected")
    
    # Check duplicates
    if anomalies.get('duplicates', {}).get('exact_duplicate_count', 0) > 0:
        alert_messages.append(f"⚠️ ALERT: {anomalies['duplicates']['exact_duplicate_count']} duplicate records found")
    
    # Check suspicious patterns
    if len(anomalies.get('suspicious_patterns', [])) > 0:
        alert_messages.append(f"⚠️ ALERT: {len(anomalies['suspicious_patterns'])} suspicious patterns detected")
    
    return '\n'.join(alert_messages) if alert_messages else "✓ No critical anomalies detected"

def detect_anomalies(input_path: str = 'data/processed/clean_reviews.csv') -> Dict[str, Any]:
    """Main anomaly detection pipeline"""
    try:
        # Load data
        logger.info(f"Loading data for anomaly detection from {input_path}")
        df = pd.read_csv(input_path)
        
        anomalies = {}
        
        # Detect outliers in numerical columns
        numerical_columns = ['useful_votes', 'funny_votes', 'cool_votes', 'text_length']
        for col in numerical_columns:
            if col in df.columns:
                anomalies[f'{col}_outliers_iqr'] = detect_outliers_iqr(df, col)
                anomalies[f'{col}_outliers_zscore'] = detect_outliers_zscore(df, col)
        
        # Detect duplicates
        anomalies['duplicates'] = detect_duplicates(df)
        
        # Detect suspicious patterns
        anomalies['suspicious_patterns'] = detect_suspicious_patterns(df)
        
        # Detect missing patterns
        anomalies['missing_patterns'] = detect_missing_patterns(df)
        
        # Generate alerts
        alert_message = generate_anomaly_alert(anomalies)
        logger.info(alert_message)
        
        # Log summary
        logger.info(f"Anomaly detection complete")
        logger.info(f"Total outliers detected: {sum(len(v) for k, v in anomalies.items() if 'outliers' in k)}")
        logger.info(f"Duplicate records: {anomalies.get('duplicates', {}).get('exact_duplicate_count', 0)}")
        logger.info(f"Suspicious patterns: {len(anomalies.get('suspicious_patterns', []))}")
        
        # If critical anomalies, log warning
        if "ALERT" in alert_message:
            logger.warning(alert_message)
            # In production, this could trigger email/Slack notification
        
        return anomalies
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise

if __name__ == "__main__":
    detect_anomalies()
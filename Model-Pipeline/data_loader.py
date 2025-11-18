"""
Data loader module for EchoAI ML Pipeline
Handles data loading, splitting, and preparation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load preprocessed data from pipeline"""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        logger.info(f"Loaded {len(df)} reviews from {PROCESSED_DATA_PATH}")
        
        # Basic data validation
        required_columns = ['review_id', 'text', 'rating', 'sentiment_label', 
                          'business_category', 'platform', 'has_owner_response']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert sentiment to numeric if needed
        if df['sentiment_label'].dtype == 'object':
            sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            df['sentiment_numeric'] = df['sentiment_label'].map(sentiment_map)
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_train_val_test_split(df, text_col='text', label_col='sentiment_numeric'):
    """
    Split data into train, validation, and test sets
    Ensures stratification and saves split indices for reproducibility
    """
    # Get features and labels
    X = df[text_col].values
    y = df[label_col].values if label_col in df.columns else df['sentiment_label'].values
    
    # Also keep metadata for bias analysis
    metadata = df[BIAS_SLICES].copy()
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, metadata,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=val_size_adjusted,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Log class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Training class distribution: {dict(zip(unique, counts))}")
    
    return {
        'train': (X_train, y_train, meta_train),
        'val': (X_val, y_val, meta_val),
        'test': (X_test, y_test, meta_test)
    }

def get_data_stats(df):
    """Get basic statistics about the dataset"""
    stats = {
        'total_reviews': len(df),
        'unique_businesses': df['business_id'].nunique() if 'business_id' in df.columns else None,
        'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else None,
        'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
        'category_distribution': df['business_category'].value_counts().to_dict(),
        'platform_distribution': df['platform'].value_counts().to_dict(),
        'avg_text_length': df['text_length'].mean() if 'text_length' in df.columns else None,
        'rating_distribution': df['rating'].value_counts().to_dict()
    }
    return stats

def prepare_data_for_training():
    """Main function to prepare data for model training"""
    logger.info("="*60)
    logger.info("Starting data preparation for model training")
    
    # Load data
    df = load_processed_data()
    
    # Get and log statistics
    stats = get_data_stats(df)
    logger.info(f"Dataset statistics: {stats}")
    
    # Create splits
    data_splits = create_train_val_test_split(df)
    
    # Save split indices for reproducibility
    split_info = {
        'train_size': len(data_splits['train'][0]),
        'val_size': len(data_splits['val'][0]),
        'test_size': len(data_splits['test'][0]),
        'random_state': RANDOM_STATE
    }
    
    import json
    with open(RESULTS_DIR / 'data_splits.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info("Data preparation complete")
    return data_splits, stats

if __name__ == "__main__":
    data_splits, stats = prepare_data_for_training()
    print("\nData preparation successful!")
    print(f"Training samples: {len(data_splits['train'][0])}")
    print(f"Validation samples: {len(data_splits['val'][0])}")
    print(f"Test samples: {len(data_splits['test'][0])}")
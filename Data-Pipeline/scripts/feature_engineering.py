# Data-Pipeline/scripts/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features"""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def extract_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sentiment-related features"""
    # Sentiment indicators based on rating
    df['is_positive'] = (df['rating'] >= 4).astype(int)
    df['is_negative'] = (df['rating'] <= 2).astype(int)
    df['is_neutral'] = (df['rating'] == 3).astype(int)
    
    # Text-based sentiment indicators
    positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing', 'poor']
    
    df['positive_word_count'] = df['cleaned_text'].apply(
        lambda x: sum(word in x.lower() for word in positive_words)
    )
    df['negative_word_count'] = df['cleaned_text'].apply(
        lambda x: sum(word in x.lower() for word in negative_words)
    )
    
    # Sentiment score
    df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']
    
    return df

def extract_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract business-related features"""
    if 'business_id' in df.columns:
        # Calculate business-level statistics
        business_stats = df.groupby('business_id').agg({
            'rating': ['mean', 'std', 'count'],
            'useful_votes': 'sum'
        }).reset_index()
        
        business_stats.columns = ['business_id', 'business_avg_rating', 
                                 'business_rating_std', 'business_review_count',
                                 'business_total_useful_votes']
        
        # Merge back to main dataframe
        df = df.merge(business_stats, on='business_id', how='left')
    
    return df

def create_features(input_path: str = 'data/processed/clean_reviews.csv',
                   output_path: str = 'data/processed/features.csv') -> pd.DataFrame:
    """Main feature engineering pipeline"""
    try:
        # Load preprocessed data
        logger.info(f"Loading preprocessed data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Extract features
        logger.info("Extracting temporal features...")
        df = extract_temporal_features(df)
        
        logger.info("Extracting sentiment features...")
        df = extract_sentiment_features(df)
        
        logger.info("Extracting business features...")
        df = extract_business_features(df)
        
        # Create interaction features
        df['rating_length_interaction'] = df['rating'] * df['text_length']
        df['votes_to_length_ratio'] = df['useful_votes'] / (df['text_length'] + 1)
        
        # Save features
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records with {len(df.columns)} features to {output_path}")
        
        # Log feature statistics
        logger.info(f"Feature dimensions: {df.shape}")
        logger.info(f"New features created: {len(df.columns) - 15}")  # Assuming ~15 original columns
        
        return df
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    create_features()
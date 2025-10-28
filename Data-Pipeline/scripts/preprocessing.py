import pandas as pd
import numpy as np
import re
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize review text"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = ' '.join(text.split())
    
    return text.strip()

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    if 'text' in df.columns:
        df['text'] = df['text'].fillna('')
    
    if 'rating' in df.columns:
        df['rating'] = df['rating'].fillna(df['rating'].median())
    
    vote_columns = ['useful_votes', 'funny_votes', 'cool_votes']
    for col in vote_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if 'review_id' in df.columns:
        df = df.dropna(subset=['review_id'])
    
    return df

def preprocess_data(input_path='data/raw/synthetic_reviews.csv',
                   output_path='data/processed/clean_reviews.csv'):
    """Main preprocessing pipeline"""
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} reviews")
        
        df = handle_missing_values(df)
        
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        df = df[df['cleaned_text'].str.len() > 0]
        
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} processed reviews to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()

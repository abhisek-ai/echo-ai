# Data-Pipeline/scripts/data_acquisition.py
import pandas as pd
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def acquire_data(source='local'):
    """
    Acquire review data from specified source
    For production: would connect to APIs
    For assignment: uses synthetic data
    """
    try:
        if source == 'local':
            # Use generated synthetic data
            filepath = 'data/raw/synthetic_reviews.csv'
            
            if not os.path.exists(filepath):
                logger.warning(f"Data file not found. Generating synthetic data...")
                from generate_data import generate_synthetic_reviews, save_data
                df = generate_synthetic_reviews(5000)
                save_data(df)
            
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} reviews from {filepath}")
            
        elif source == 'api':
            # Placeholder for real API integration
            logger.info("API integration would go here in production")
            df = pd.DataFrame()  # Would fetch from Google Reviews API
            
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Data quality checks
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Data acquisition failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = acquire_data()
    print(f"Successfully acquired {len(df)} reviews")
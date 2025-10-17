import pandas as pd
import os

def fetch_reviews():
    """Fetch review data from source"""
    input_path = '/opt/airflow/data/raw/sample_reviews.csv'
    df = pd.read_csv(input_path)
    print(f"Fetched {len(df)} reviews")
    return df

if __name__ == "__main__":
    fetch_reviews()

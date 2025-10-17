import pandas as pd
import re

def clean_text(text):
    """Basic text cleaning"""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    return text.lower().strip()

def preprocess_reviews():
    """Preprocess review data"""
    df = pd.read_csv('/opt/airflow/data/raw/sample_reviews.csv')
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Add sentiment labels
    df['sentiment'] = df['rating'].apply(
        lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
    )
    
    # Save processed data
    df.to_csv('/opt/airflow/data/processed/clean_reviews.csv', index=False)
    print(f"Processed {len(df)} reviews")
    return df

if __name__ == "__main__":
    preprocess_reviews()

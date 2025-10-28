# Data-Pipeline/scripts/generate_data.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_reviews(n=5000):
    """Generate realistic synthetic review data for EchoAI"""
    
    # Realistic review templates by rating
    templates = {
        5: [
            "Absolutely amazing {item}! The {aspect} was {positive_adj} and {positive_adj2}.",
            "Best {item} I've ever experienced. {aspect} exceeded all expectations!",
            "Incredible {aspect}! Will definitely recommend to everyone.",
            "Perfect in every way. The {item} was {positive_adj} beyond words."
        ],
        4: [
            "Really good {item}. The {aspect} was {positive_adj}.",
            "Great experience overall. Minor issues but {aspect} was {positive_adj}.",
            "Very satisfied with the {item}. {aspect} could be slightly better.",
            "Good {item}, especially the {aspect}. Would come again."
        ],
        3: [
            "The {item} was okay. {aspect} was average.",
            "Nothing special but decent {item}. {aspect} could improve.",
            "Mixed experience. Some parts good, {aspect} was mediocre.",
            "It was fine. The {item} met basic expectations."
        ],
        2: [
            "Disappointed with the {item}. {aspect} was {negative_adj}.",
            "Not worth it. The {aspect} was particularly {negative_adj}.",
            "Below expectations. {item} needs major improvements.",
            "Poor {aspect}. Would not recommend this {item}."
        ],
        1: [
            "Terrible {item}! {aspect} was absolutely {negative_adj}.",
            "Worst experience ever. The {aspect} was {negative_adj} and {negative_adj2}.",
            "Complete disaster. Everything about the {item} was awful.",
            "Unacceptable {aspect}. Never returning."
        ]
    }
    
    # Word banks
    items = ['service', 'food', 'product', 'experience', 'facility']
    aspects = ['quality', 'staff', 'atmosphere', 'cleanliness', 'value', 'timing']
    positive_adj = ['excellent', 'outstanding', 'fantastic', 'wonderful', 'exceptional', 'great']
    negative_adj = ['terrible', 'awful', 'poor', 'disappointing', 'unacceptable', 'horrible']
    
    # Business categories for bias analysis
    categories = ['Restaurant', 'Hotel', 'Retail', 'Healthcare', 'Automotive']
    
    reviews = []
    
    # Generate reviews with realistic distribution
    rating_distribution = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    
    for i in range(n):
        rating = random.choice(rating_distribution)
        template = random.choice(templates[rating])
        
        # Fill in template
        text = template.format(
            item=random.choice(items),
            aspect=random.choice(aspects),
            positive_adj=random.choice(positive_adj),
            positive_adj2=random.choice(positive_adj),
            negative_adj=random.choice(negative_adj),
            negative_adj2=random.choice(negative_adj)
        )
        
        # Add realistic metadata
        review_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        reviews.append({
            'review_id': f'R{i:06d}',
            'business_id': f'B{random.randint(1, 100):04d}',
            'user_id': f'U{random.randint(1, 1000):05d}',
            'text': text,
            'rating': rating,
            'date': review_date.strftime('%Y-%m-%d'),
            'time': review_date.strftime('%H:%M:%S'),
            'useful_votes': np.random.poisson(2),  # Poisson distribution for votes
            'funny_votes': np.random.poisson(0.5),
            'cool_votes': np.random.poisson(1),
            'text_length': len(text),
            'business_category': random.choice(categories),
            'response_time_hours': random.randint(1, 168) if random.random() > 0.7 else None,
            'has_owner_response': random.random() > 0.7,
            'verified_purchase': random.random() > 0.2,
            'platform': random.choice(['Google', 'Yelp', 'Facebook', 'TripAdvisor']),
            'language': 'en',
            'sentiment_label': 'positive' if rating >= 4 else 'negative' if rating <= 2 else 'neutral'
        })
    
    df = pd.DataFrame(reviews)
    
    # Add some data quality issues for testing
    # Add 1% missing values
    missing_indices = random.sample(range(len(df)), int(0.01 * len(df)))
    df.loc[missing_indices, 'text'] = np.nan
    
    # Add some outliers
    outlier_indices = random.sample(range(len(df)), int(0.02 * len(df)))
    df.loc[outlier_indices, 'useful_votes'] = random.randint(100, 500)
    
    return df

def save_data(df, filepath='data/raw/synthetic_reviews.csv'):
    """Save generated data with logging"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Generated {len(df)} reviews")
    print(f"✓ Rating distribution: {df['rating'].value_counts().sort_index().to_dict()}")
    print(f"✓ Missing values: {df.isnull().sum().sum()}")
    print(f"✓ Saved to: {filepath}")
    
    # Create a sample for quick testing
    df.head(100).to_csv('data/raw/sample_reviews.csv', index=False)
    print(f"✓ Sample (100 rows) saved to: data/raw/sample_reviews.csv")

if __name__ == "__main__":
    df = generate_synthetic_reviews(5000)
    save_data(df)
    
    # Generate statistics
    print("\nDataset Statistics:")
    print(f"- Total reviews: {len(df)}")
    print(f"- Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"- Unique businesses: {df['business_id'].nunique()}")
    print(f"- Unique users: {df['user_id'].nunique()}")
    print(f"- Average text length: {df['text_length'].mean():.0f} characters")
    print(f"- Response rate: {df['has_owner_response'].mean():.1%}")
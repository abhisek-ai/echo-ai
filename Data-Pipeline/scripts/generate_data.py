import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_reviews(n=5000):
    """Generate realistic synthetic review data for EchoAI"""
    
    templates = {
        5: ["Absolutely amazing {item}! The {aspect} was excellent.",
            "Best experience ever. Highly recommend!",
            "Perfect in every way. Outstanding service."],
        4: ["Really good {item}. The {aspect} was great.",
            "Very satisfied. Would come again.",
            "Good experience overall."],
        3: ["The {item} was okay. Average {aspect}.",
            "Nothing special but decent.",
            "It was fine."],
        2: ["Disappointed with the {item}. Poor {aspect}.",
            "Not worth it. Below expectations.",
            "Would not recommend."],
        1: ["Terrible {item}! Awful {aspect}.",
            "Worst experience ever.",
            "Complete disaster."]
    }
    
    items = ['service', 'food', 'product', 'experience']
    aspects = ['quality', 'staff', 'atmosphere', 'value']
    categories = ['Restaurant', 'Hotel', 'Retail', 'Healthcare', 'Automotive']
    
    reviews = []
    rating_distribution = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    
    for i in range(n):
        rating = random.choice(rating_distribution)
        template = random.choice(templates[rating])
        
        text = template.format(
            item=random.choice(items),
            aspect=random.choice(aspects)
        )
        
        review_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        reviews.append({
            'review_id': f'R{i:06d}',
            'business_id': f'B{random.randint(1, 100):04d}',
            'user_id': f'U{random.randint(1, 1000):05d}',
            'text': text,
            'rating': rating,
            'date': review_date.strftime('%Y-%m-%d'),
            'useful_votes': np.random.poisson(2),
            'funny_votes': np.random.poisson(0.5),
            'cool_votes': np.random.poisson(1),
            'text_length': len(text),
            'business_category': random.choice(categories),
            'has_owner_response': random.random() > 0.7,
            'platform': random.choice(['Google', 'Yelp', 'Facebook']),
            'sentiment_label': 'positive' if rating >= 4 else 'negative' if rating <= 2 else 'neutral'
        })
    
    df = pd.DataFrame(reviews)
    
    # Add some missing values for testing
    missing_indices = random.sample(range(len(df)), int(0.01 * len(df)))
    df.loc[missing_indices, 'text'] = np.nan
    
    return df

def save_data(df, filepath='data/raw/synthetic_reviews.csv'):
    """Save generated data"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Generated {len(df)} reviews")
    print(f"✓ Saved to: {filepath}")

if __name__ == "__main__":
    df = generate_synthetic_reviews(5000)
    save_data(df)

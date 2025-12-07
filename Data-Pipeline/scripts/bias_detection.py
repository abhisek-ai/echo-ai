import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FEATURES_FILE = PROCESSED_DIR / "features.csv"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
BIAS_REPORT_FILE = DOCS_DIR / "bias_report.md"

def analyze_rating_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    rating_dist = df['reviewRating'].value_counts(normalize=True).sort_index()
    analysis = {
        'distribution': rating_dist.to_dict(),
        'mean_rating': df['reviewRating'].mean(),
        'median_rating': df['reviewRating'].median(),
        'std_rating': df['reviewRating'].std(),
        'skewness': df['reviewRating'].skew(),
        'positive_ratio': (df['reviewRating'] >= 4).mean(),
        'negative_ratio': (df['reviewRating'] <= 2).mean(),
        'neutral_ratio': (df['reviewRating'] == 3).mean()
    }
    # Check for bias
    if analysis['positive_ratio'] > 0.7:
        analysis['bias_detected'] = 'Positive bias - over 70% positive reviews'
    elif analysis['negative_ratio'] > 0.5:
        analysis['bias_detected'] = 'Negative bias - over 50% negative reviews'
    else:
        analysis['bias_detected'] = 'No significant bias detected'
    return analysis

def analyze_categorical_bias(df: pd.DataFrame, category_col: str) -> Dict[str, Any]:
    if category_col not in df.columns:
        return {}
    category_analysis = {}
    for cat in df[category_col].unique():
        subset = df[df[category_col]==cat]
        category_analysis[str(cat)] = {
            'count': len(subset),
            'percentage': (len(subset)/len(df))*100,
            'avg_rating': subset['reviewRating'].mean(),
            'rating_std': subset['reviewRating'].std()
        }
    counts = [v['count'] for v in category_analysis.values()]
    imbalance_ratio = max(counts)/min(counts) if min(counts)>0 else float('inf')
    return {
        'categories': category_analysis,
        'imbalance_ratio': imbalance_ratio,
        'most_common': max(category_analysis.items(), key=lambda x:x[1]['count'])[0],
        'least_common': min(category_analysis.items(), key=lambda x:x[1]['count'])[0]
    }

def analyze_temporal_bias(df: pd.DataFrame) -> Dict[str, Any]:
    if 'reviewDate' not in df.columns:
        return {}
    df['reviewDate'] = pd.to_datetime(df['reviewDate'])
    df['month'] = df['reviewDate'].dt.to_period('M')
    monthly_trend = df.groupby('month').agg({'reviewRating':['mean','std','count']})
    return {
        'monthly_trend': monthly_trend.to_dict(),
        'trend_description': 'Temporal analysis completed'
    }

def analyze_text_length_bias(df: pd.DataFrame) -> Dict[str, Any]:
    if "text_length" not in df.columns:
        return {}
    short = df[df["text_length"]<20]
    long = df[df["text_length"]>df["text_length"].quantile(0.9)]
    return {
        "short_reviews_count": len(short),
        "long_reviews_count": len(long),
        "avg_rating_short": short["reviewRating"].mean() if len(short) else None,
        "avg_rating_long": long["reviewRating"].mean() if len(long) else None,
        "bias_hint": "Short reviews may indicate spam/low-effort content"
    }

def detect_and_report_bias(input_path: Path = FEATURES_FILE) -> Dict[str, Any]:
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        results = {
            "rating_distribution": analyze_rating_distribution(df),
            "sentiment_bias": analyze_categorical_bias(df, "sentiment"),
            "restaurant_bias": analyze_categorical_bias(df, "placeName"),
            "text_length_bias": analyze_text_length_bias(df)
        }
        if 'reviewDate' in df.columns:
            results["temporal_bias"] = analyze_temporal_bias(df)

        # Save Markdown report
        with open(BIAS_REPORT_FILE, "w") as f:
            f.write("# Bias Detection Report\n\n")
            for section, data in results.items():
                f.write(f"## {section}\n")
                f.write(str(data)+"\n\n")

        logger.info(f"Bias detection complete. Report saved to {BIAS_REPORT_FILE}")
        return results

    except Exception as e:
        logger.error(f"Bias detection failed: {e}")
        raise

if __name__ == "__main__":
    detect_and_report_bias()

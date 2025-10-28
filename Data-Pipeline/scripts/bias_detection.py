# Data-Pipeline/scripts/bias_detection.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_rating_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze bias in rating distribution"""
    rating_dist = df['rating'].value_counts(normalize=True).sort_index()
    
    analysis = {
        'distribution': rating_dist.to_dict(),
        'mean_rating': df['rating'].mean(),
        'median_rating': df['rating'].median(),
        'std_rating': df['rating'].std(),
        'skewness': df['rating'].skew(),
        'positive_ratio': (df['rating'] >= 4).mean(),
        'negative_ratio': (df['rating'] <= 2).mean(),
        'neutral_ratio': (df['rating'] == 3).mean()
    }
    
    # Check for imbalance
    if analysis['positive_ratio'] > 0.7:
        analysis['bias_detected'] = 'Positive bias - over 70% positive reviews'
    elif analysis['negative_ratio'] > 0.5:
        analysis['bias_detected'] = 'Negative bias - over 50% negative reviews'
    else:
        analysis['bias_detected'] = 'No significant bias detected'
    
    return analysis

def analyze_categorical_bias(df: pd.DataFrame, category_col: str) -> Dict[str, Any]:
    """Analyze bias across categorical variables"""
    if category_col not in df.columns:
        return {}
    
    category_analysis = {}
    categories = df[category_col].unique()
    
    for category in categories:
        category_df = df[df[category_col] == category]
        category_analysis[str(category)] = {
            'count': len(category_df),
            'percentage': (len(category_df) / len(df)) * 100,
            'avg_rating': category_df['rating'].mean(),
            'rating_std': category_df['rating'].std()
        }
    
    # Check for category imbalance
    counts = [v['count'] for v in category_analysis.values()]
    max_count = max(counts)
    min_count = min(counts)
    
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'categories': category_analysis,
        'imbalance_ratio': imbalance_ratio,
        'most_common': max(category_analysis.items(), key=lambda x: x[1]['count'])[0],
        'least_common': min(category_analysis.items(), key=lambda x: x[1]['count'])[0]
    }

def analyze_temporal_bias(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze bias over time"""
    if 'date' not in df.columns:
        return {}
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    temporal_analysis = df.groupby('month').agg({
        'rating': ['mean', 'std', 'count']
    })
    
    return {
        'monthly_trend': temporal_analysis.to_dict(),
        'trend_description': 'Temporal analysis completed'
    }

def analyze_text_length_bias(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze bias based on text length"""
    if 'text_length' not in df.columns:
        return {}
    
    # Create text length bins
    df['length_category'] = pd.cut(df['text_length'], 
                                  bins=[0, 50, 150, 500, float('inf')],
                                  labels=['very_short', 'short', 'medium', 'long'])
    
    length_bias = df.groupby('length_category')['rating'].agg(['mean', 'count'])
    
    return {
        'length_categories': length_bias.to_dict(),
        'correlation': df['text_length'].corr(df['rating'])
    }

def generate_bias_report(bias_results: Dict[str, Any]) -> str:
    """Generate markdown report of bias analysis"""
    report = ["# Bias Detection Report\n"]
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Rating distribution
    report.append("## Rating Distribution Analysis\n")
    rating_analysis = bias_results.get('rating_distribution', {})
    report.append(f"- Mean Rating: {rating_analysis.get('mean_rating', 'N/A'):.2f}")
    report.append(f"- Positive Review Ratio: {rating_analysis.get('positive_ratio', 0):.1%}")
    report.append(f"- Negative Review Ratio: {rating_analysis.get('negative_ratio', 0):.1%}")
    report.append(f"- **Bias Status**: {rating_analysis.get('bias_detected', 'Unknown')}\n")
    
    # Category bias
    report.append("## Categorical Bias Analysis\n")
    for category_type, analysis in bias_results.items():
        if 'categories' in str(analysis):
            report.append(f"### {category_type}")
            if isinstance(analysis, dict) and 'imbalance_ratio' in analysis:
                report.append(f"- Imbalance Ratio: {analysis['imbalance_ratio']:.2f}")
                report.append(f"- Most Common: {analysis.get('most_common', 'N/A')}")
                report.append(f"- Least Common: {analysis.get('least_common', 'N/A')}\n")
    
    # Mitigation recommendations
    report.append("## Mitigation Recommendations\n")
    if rating_analysis.get('positive_ratio', 0) > 0.7:
        report.append("1. **Oversample negative reviews** to balance the dataset")
        report.append("2. **Apply class weights** during model training")
        report.append("3. **Use stratified sampling** for train/test splits")
    elif rating_analysis.get('negative_ratio', 0) > 0.5:
        report.append("1. **Oversample positive reviews** to balance the dataset")
        report.append("2. **Consider synthetic data generation** for positive examples")
    else:
        report.append("- No significant bias detected, proceed with standard training")
    
    return '\n'.join(report)

def detect_and_report_bias(input_path: str = 'data/processed/clean_reviews.csv') -> Dict[str, Any]:
    """Main bias detection pipeline"""
    try:
        # Load data
        logger.info(f"Loading data for bias detection from {input_path}")
        df = pd.read_csv(input_path)
        
        # Perform bias analysis
        bias_results = {
            'rating_distribution': analyze_rating_distribution(df),
            'text_length_bias': analyze_text_length_bias(df)
        }
        
        # Analyze categorical biases if columns exist
        if 'business_category' in df.columns:
            bias_results['business_category_bias'] = analyze_categorical_bias(df, 'business_category')
        
        if 'platform' in df.columns:
            bias_results['platform_bias'] = analyze_categorical_bias(df, 'platform')
        
        # Analyze temporal bias if date exists
        if 'date' in df.columns:
            bias_results['temporal_bias'] = analyze_temporal_bias(df)
        
        # Generate report
        report = generate_bias_report(bias_results)
        
        # Save report
        os.makedirs('docs', exist_ok=True)
        with open('docs/bias_report.md', 'w') as f:
            f.write(report)
        
        logger.info("Bias analysis complete")
        logger.info(f"Report saved to docs/bias_report.md")
        
        return bias_results
        
    except Exception as e:
        logger.error(f"Bias detection failed: {e}")
        raise

if __name__ == "__main__":
    detect_and_report_bias()
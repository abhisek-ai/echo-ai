# Data-Pipeline/tests/test_bias_detection.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts.bias_detection import analyze_rating_distribution

def test_bias_detection():
    """Test bias detection functions"""
    # Create biased dataset
    df_biased = pd.DataFrame({
        'rating': [5] * 80 + [4] * 10 + [3] * 5 + [2] * 3 + [1] * 2
    })
    
    analysis = analyze_rating_distribution(df_biased)
    assert analysis['positive_ratio'] > 0.7
    assert 'Positive bias' in analysis['bias_detected']
    print("✓ Bias detection tests passed")

def test_balanced_dataset():
    """Test with balanced dataset"""
    df_balanced = pd.DataFrame({
        'rating': [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 20
    })
    
    analysis = analyze_rating_distribution(df_balanced)
    assert analysis['positive_ratio'] == 0.4
    assert analysis['negative_ratio'] == 0.4
    print("✓ Balanced dataset tests passed")

if __name__ == "__main__":
    test_bias_detection()
    test_balanced_dataset()
    print("\n✅ All bias detection tests passed!")
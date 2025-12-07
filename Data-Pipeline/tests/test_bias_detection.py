import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from scripts.bias_detection import (
    analyze_rating_distribution,
    analyze_categorical_bias,
    analyze_temporal_bias,
    analyze_text_length_bias
)

def test_empty_dataframe_rating_distribution():
    """Empty DF should not break rating distribution analysis"""
    df = pd.DataFrame(columns=[
        "placeName","placeAddress","provider","reviewText",
        "reviewDate","reviewRating","authorName"
    ])
    result = analyze_rating_distribution(df)
    assert isinstance(result, dict)
    print("Empty dataframe rating distribution test passed")


def test_single_review_rating_distribution():
    df = pd.DataFrame({
        "reviewRating": [5]
    })
    result = analyze_rating_distribution(df)
    assert result["mean_rating"] == 5
    assert result["median_rating"] == 5
    print("Single review rating distribution test passed")


def test_categorical_bias_placeName():
    df = pd.DataFrame({
        "placeName": ["A", "B", "A", "C"],
        "rating": [5, 3, 4, 2]
    })
    result = analyze_categorical_bias(df, "placeName")
    assert "A" in result["categories"]
    assert result["imbalance_ratio"] >= 1
    print("Categorical bias (placeName) test passed")


def test_categorical_bias_missing_column():
    df = pd.DataFrame({"reviewRating": [5, 4, 3]})
    result = analyze_categorical_bias(df, "sentiment")
    assert result == {}
    print("✓ Categorical bias missing column test passed")


def test_temporal_bias_valid_dates():
    df = pd.DataFrame({
        "reviewDate": ["2024-01-01", "2024-01-15", "2024-02-01"],
        "rating": [5, 4, 3]
    })
    result = analyze_temporal_bias(df)
    assert "monthly_trend" in result
    print("Temporal bias with valid dates test passed")


def test_temporal_bias_missing_column():
    df = pd.DataFrame({"rating": [4, 3, 5]})
    result = analyze_temporal_bias(df)
    assert result == {}
    print("✓ Temporal bias missing column test passed")

def test_text_length_bias_basic():
    df = pd.DataFrame({
        "text_length": [5, 10, 100, 150],
        "reviewRating": [3, 4, 5, 2]
    })
    result = analyze_text_length_bias(df)
    assert "short_reviews_count" in result
    assert "long_reviews_count" in result
    print("✓ Text length bias basic test passed")


def test_text_length_bias_missing_column():
    df = pd.DataFrame({"reviewRating": [1, 2, 3]})
    result = analyze_text_length_bias(df)
    assert result == {}
    print("✓ Text length bias missing column test passed")


if __name__ == "__main__":
    test_empty_dataframe_rating_distribution()
    test_single_review_rating_distribution()
    test_categorical_bias_placeName()
    test_categorical_bias_missing_column()
    test_temporal_bias_valid_dates()
    test_temporal_bias_missing_column()
    test_text_length_bias_basic()
    test_text_length_bias_missing_column()

    print("\n All bias detection edge case tests passed!")

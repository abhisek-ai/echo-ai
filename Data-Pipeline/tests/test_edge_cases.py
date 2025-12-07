# Data-Pipeline/tests/test_bias_edge_cases.py

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

def test_empty_rating_distribution():
    """Test rating distribution with empty dataframe"""
    df = pd.DataFrame(columns=["reviewRating"])
    result = analyze_rating_distribution(df)
    assert isinstance(result, dict)
    print(" Empty rating distribution test passed")


def test_single_rating():
    """Test with a single review rating"""
    df = pd.DataFrame({"reviewRating": [5]})
    result = analyze_rating_distribution(df)
    assert result["mean_rating"] == 5
    assert result["median_rating"] == 5
    print(" Single rating test passed")



def test_categorical_missing_column():
    """Test categorical bias when column does not exist"""
    df = pd.DataFrame({"reviewRating": [5, 4, 3]})
    result = analyze_categorical_bias(df, "placeName")
    assert result == {}
    print(" Missing categorical column test passed")


def test_categorical_basic():
    """Test simple categorical bias detection"""
    df = pd.DataFrame({
        "placeName": ["A", "A", "B"],
        "rating": [5, 4, 3]
    })
    result = analyze_categorical_bias(df, "placeName")
    assert "A" in result["categories"]
    assert "B" in result["categories"]
    print(" Basic categorical bias test passed")

def test_temporal_missing_column():
    """Test temporal bias when reviewDate does not exist"""
    df = pd.DataFrame({"reviewRating": [5, 4, 3]})
    result = analyze_temporal_bias(df)
    assert result == {}
    print(" Temporal bias missing column test passed")


def test_temporal_simple():
    """Test temporal bias with valid dates"""
    df = pd.DataFrame({
        "reviewDate": ["2024-01-01", "2024-01-15", "2024-02-01"],
        "rating": [5, 4, 3]
    })
    result = analyze_temporal_bias(df)
    assert "monthly_trend" in result
    print(" Basic temporal bias test passed")



def test_text_length_missing_column():
    """Test text-length bias when column is missing"""
    df = pd.DataFrame({"reviewRating": [5, 4, 3]})
    result = analyze_text_length_bias(df)
    assert result == {}
    print(" Missing text_length column test passed")


def test_text_length_basic():
    """Test text length bias with short and long reviews"""
    df = pd.DataFrame({
        "text_length": [5, 10, 200],
        "reviewRating": [3, 4, 5]
    })
    result = analyze_text_length_bias(df)
    assert "short_reviews_count" in result
    assert "long_reviews_count" in result
    print(" Basic text length bias test passed")


if __name__ == "__main__":
    test_empty_rating_distribution()
    test_single_rating()

    test_categorical_missing_column()
    test_categorical_basic()

    test_temporal_missing_column()
    test_temporal_simple()

    test_text_length_missing_column()
    test_text_length_basic()

    print("\n All bias detection edge case tests passed!")

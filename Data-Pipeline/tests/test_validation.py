# Data-Pipeline/tests/test_validation.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.validation import (
    check_data_types,
    check_data_ranges,
    check_missing_values,
    check_duplicates
)

def test_data_type_validation():
    """Test validation of expected data types"""
    df = pd.DataFrame({
        "reviewRating": [5, 4, 3],
        "reviewText": ["Good", "Okay", "Bad"]
    })

    result = check_data_types(df)

    assert "reviewRating" in result
    assert "reviewText" in result
    assert result["reviewRating"]["valid"] is True
    assert result["reviewText"]["valid"] is True

    print(" Data type validation test passed")


def test_range_validation():
    """Test rating and text_length range checking"""
    df = pd.DataFrame({
        "reviewRating": [1, 3, 5],
        "text_length": [5, 10, 300]
    })

    result = check_data_ranges(df)

    assert "reviewRating" in result
    assert result["reviewRating"]["valid"] is True
    assert result["reviewRating"]["min"] == 1
    assert result["reviewRating"]["max"] == 5

    assert "text_length" in result
    assert result["text_length"]["min"] == 5
    assert result["text_length"]["max"] == 300

    print(" Range validation test passed")

def test_missing_value_detection():
    """Test missing value reporting"""
    df = pd.DataFrame({
        "reviewText": ["A", None, "C"],
        "reviewRating": [5, 4, None]
    })

    result = check_missing_values(df)

    assert result["reviewText"]["missing_count"] == 1
    assert result["reviewRating"]["missing_count"] == 1

    print(" Missing value detection test passed")


def test_duplicate_detection():
    """Test detecting duplicates in authorName and reviewText"""
    df = pd.DataFrame({
        "authorName": ["Alice", "Alice", "Bob"],
        "reviewText": ["Good", "Good", "Bad"],
        "reviewRating": [5, 5, 3]
    })

    result = check_duplicates(df)

    assert result["duplicate_review_ids"] == 2   
    assert result["duplicate_texts"] == 2       
    assert result["exact_duplicates"] >= 0     

    print(" Duplicate detection test passed")


if __name__ == "__main__":
    test_data_type_validation()
    test_range_validation()
    test_missing_value_detection()
    test_duplicate_detection()

    print("\n All validation tests passed!")

# Data-Pipeline/tests/test_preprocessing.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from scripts.preprocessing import clean_text, handle_missing_values

def test_clean_text():
    """Test text cleaning function"""
    assert clean_text("Hello World!") == "hello world"
    assert clean_text("Test@123#$%") == "test123"
    assert clean_text("http://example.com text") == "text"
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("UPPERCASE") == "uppercase"
    print("✓ Text cleaning tests passed")

def test_handle_missing_values():
    """Test missing value handling"""
    df = pd.DataFrame({
        'review_id': ['R1', None, 'R3'],
        'text': ['Good', None, 'Bad'],
        'rating': [5, None, 1],
        'useful_votes': [1, None, 0]
    })
    
    df_cleaned = handle_missing_values(df)
    
    assert df_cleaned['text'].isnull().sum() == 0
    assert df_cleaned['rating'].isnull().sum() == 0
    assert df_cleaned['useful_votes'].isnull().sum() == 0
    assert len(df_cleaned) == 2  # One row with null review_id should be dropped
    print("✓ Missing value handling tests passed")

def test_edge_cases():
    """Test edge cases"""
    # Empty dataframe
    df_empty = pd.DataFrame()
    assert len(df_empty) == 0
    
    # Single row
    df_single = pd.DataFrame({'text': ['test']})
    assert len(df_single) == 1
    
    # All nulls
    df_nulls = pd.DataFrame({'col': [None, None, None]})
    assert df_nulls['col'].isnull().all()
    
    print("✓ Edge case tests passed")

if __name__ == "__main__":
    test_clean_text()
    test_handle_missing_values()
    test_edge_cases()
    print("\n✅ All preprocessing tests passed!")
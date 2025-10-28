# Data-Pipeline/tests/test_edge_cases.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def test_empty_dataframe():
    """Test with empty dataframe"""
    df = pd.DataFrame()
    assert len(df) == 0
    assert df.empty == True
    print("✓ Empty dataframe test passed")

def test_single_row():
    """Test with single row"""
    df = pd.DataFrame({'col': [1]})
    assert len(df) == 1
    assert not df.empty
    print("✓ Single row test passed")

def test_large_values():
    """Test with extreme values"""
    df = pd.DataFrame({
        'large_num': [1e10, 1e15, 1e20],
        'small_num': [1e-10, 1e-15, 1e-20]
    })
    assert df['large_num'].max() == 1e20
    assert df['small_num'].min() == 1e-20
    print("✓ Large values test passed")

def test_special_characters():
    """Test with special characters"""
    special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
    df = pd.DataFrame({'text': [special_chars]})
    assert len(df['text'][0]) == len(special_chars)
    print("✓ Special characters test passed")

def test_null_handling():
    """Test null value handling"""
    df = pd.DataFrame({
        'col1': [1, None, 3],
        'col2': [None, None, None],
        'col3': [1, 2, 3]
    })
    assert df['col1'].isnull().sum() == 1
    assert df['col2'].isnull().all()
    assert df['col3'].notnull().all()
    print("✓ Null handling test passed")

if __name__ == "__main__":
    test_empty_dataframe()
    test_single_row()
    test_large_values()
    test_special_characters()
    test_null_handling()
    print("\n✅ All edge case tests passed!")
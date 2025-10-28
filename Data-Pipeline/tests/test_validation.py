# Data-Pipeline/tests/test_validation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from scripts.validation import check_data_types, check_data_ranges

def test_data_type_validation():
    """Test data type checking"""
    df = pd.DataFrame({
        'review_id': ['R1', 'R2'],
        'rating': [5, 4],
        'text': ['Good', 'Great']
    })
    
    type_results = check_data_types(df)
    assert 'review_id' in type_results
    assert type_results['review_id']['valid'] == True
    print("✓ Data type validation tests passed")

def test_range_validation():
    """Test range checking"""
    df = pd.DataFrame({
        'rating': [1, 2, 3, 4, 5],
        'useful_votes': [0, 1, 2, 3, 4],
        'text_length': [10, 20, 30, 40, 50]
    })
    
    range_results = check_data_ranges(df)
    assert range_results['rating']['valid'] == True
    assert range_results['rating']['min'] == 1
    assert range_results['rating']['max'] == 5
    print("✓ Range validation tests passed")

if __name__ == "__main__":
    test_data_type_validation()
    test_range_validation()
    print("\n✅ All validation tests passed!")
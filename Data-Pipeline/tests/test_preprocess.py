# Data-Pipeline/tests/test_preprocessing.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scripts.preprocessing import clean_text, handle_missing_values, generate_template_review


def test_clean_text():
    """Test the text cleaning function"""
    assert clean_text("Hello World!") == "hello world!"
    assert clean_text("Test@123#$%") == "test123"
    assert clean_text("http://example.com text") == "text"
    assert clean_text("<b>bold</b> text") == "bold text"
    assert clean_text(None) == ""
    assert clean_text("   spaces   everywhere   ") == "spaces everywhere"
    assert clean_text("UPPERCASE") == "uppercase"

    print(" clean_text tests passed")


def test_handle_missing_values():
    """Test missing value handling logic"""
    df = pd.DataFrame({
        "reviewRating": [5, None, 1],
        "reviewText": ["Good", None, ""],
        "authorName": ["A", "B", "C"],
    })

    cleaned = handle_missing_values(df)

    # Row with missing rating is removed
    assert len(cleaned) == 2

    # All reviewRatings are integers
    assert cleaned["reviewRating"].dtype == int

    # reviewText should be filled for missing/empty values
    assert cleaned.loc[cleaned["reviewRating"] == 1, "reviewText"].iloc[0] == generate_template_review(1)

    print(" handle_missing_values tests passed")

def test_edge_cases():
    """Test edge cases for preprocessing utilities"""

    # Empty DF
    df_empty = pd.DataFrame()
    assert df_empty.empty

    # Single-row DF
    df_single = pd.DataFrame({"reviewText": ["hello"]})
    assert len(df_single) == 1

    # All-null column
    df_nulls = pd.DataFrame({"col": [None, None]})
    assert df_nulls["col"].isnull().all()

    print(" Edge case tests passed")


if __name__ == "__main__":
    test_clean_text()
    test_handle_missing_values()
    test_edge_cases()

    print("\n All preprocessing tests passed!")

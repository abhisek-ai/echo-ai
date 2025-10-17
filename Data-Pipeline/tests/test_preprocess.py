import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess import clean_text

def test_clean_text():
    assert clean_text("Hello World!") == "hello world"
    assert clean_text("Test@123") == "test"
    assert clean_text("") == ""

def test_text_with_numbers():
    assert clean_text("Review 123") == "review "

if __name__ == "__main__":
    test_clean_text()
    test_text_with_numbers()
    print("All tests passed!")

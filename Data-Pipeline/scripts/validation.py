import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FEATURES_FILE = PROCESSED_DIR / "features.csv"
METRICS_DIR = PROJECT_ROOT / "data/metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_OUTPUT = METRICS_DIR / "validation_results.json"

def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data types"""
    expected_types = {
        'reviewRating': 'int64',
        'reviewText': 'object',
        'text_length': 'int64',
        'restaurant_avg_rating': 'float64',
        'sentiment': 'object'
    }
    
    type_check_results = {}
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            type_check_results[col] = {
                'expected': expected_type,
                'actual': actual_type,
                'valid': expected_type in actual_type
            }
        else:
            type_check_results[col] = {
                'expected': expected_type,
                'actual': None,
                'valid': False
            }
    return type_check_results

def check_data_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate numeric ranges"""
    range_checks = {}
    if 'reviewRating' in df.columns:
        range_checks['reviewRating'] = {
            'min': int(df['reviewRating'].min()),
            'max': int(df['reviewRating'].max()),
            'valid': df['reviewRating'].between(1,5).all()
        }
    if 'text_length' in df.columns:
        range_checks['text_length'] = {
            'min': int(df['text_length'].min()),
            'max': int(df['text_length'].max()),
            'mean': float(df['text_length'].mean()),
            'valid': df['text_length'].min() >= 0
        }
    return range_checks

def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for missing values"""
    missing_report = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_report[col] = {
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_pct, 2),
            'is_critical': col in ['reviewText', 'reviewRating']
        }
    return missing_report

def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicates"""
    duplicate_report = {
        'duplicate_authorName': int(df.duplicated(subset=['authorName'], keep=False).sum()),
        'duplicate_reviewText': int(df.duplicated(subset=['reviewText'], keep=False).sum()),
        'exact_duplicates': int(df.duplicated(keep=False).sum())
    }
    return duplicate_report

def validate_data(input_path: Path = FEATURES_FILE) -> Dict[str, Any]:
    """Main validation pipeline"""
    try:
        logger.info(f"Loading data from {input_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found")

        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Run validation checks
        type_checks = check_data_types(df)
        range_checks = check_data_ranges(df)
        missing_checks = check_missing_values(df)
        duplicate_checks = check_duplicates(df)

        # Determine overall validation
        validation_passed = all([v['valid'] for v in type_checks.values()]) \
                            and all([v['valid'] for v in range_checks.values()]) \
                            and all([v['missing_count']==0 for k,v in missing_checks.items() if v['is_critical']])

        validation_results = {
            "validation_passed": validation_passed,
            "type_checks": type_checks,
            "range_checks": range_checks,
            "missing_checks": missing_checks,
            "duplicate_checks": duplicate_checks,
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }

        # Save validation results
        with open(VALIDATION_OUTPUT, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)

        logger.info(f"Validation complete. Passed: {validation_passed}")
        logger.info(f"Results saved to {VALIDATION_OUTPUT}")

        return validation_results

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    validate_data()

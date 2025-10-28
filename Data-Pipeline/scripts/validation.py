# Data-Pipeline/scripts/validation.py
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data types"""
    expected_types = {
        'review_id': 'object',
        'rating': 'int64',
        'text': 'object',
        'useful_votes': 'int64'
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
    
    return type_check_results

def check_data_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data ranges"""
    range_checks = {}
    
    # Check rating range
    if 'rating' in df.columns:
        range_checks['rating'] = {
            'min': df['rating'].min(),
            'max': df['rating'].max(),
            'valid': (df['rating'].min() >= 1) and (df['rating'].max() <= 5),
            'out_of_range_count': len(df[(df['rating'] < 1) | (df['rating'] > 5)])
        }
    
    # Check text length
    if 'text_length' in df.columns:
        range_checks['text_length'] = {
            'min': df['text_length'].min(),
            'max': df['text_length'].max(),
            'mean': df['text_length'].mean(),
            'empty_count': len(df[df['text_length'] == 0])
        }
    
    # Check votes are non-negative
    if 'useful_votes' in df.columns:
        range_checks['useful_votes'] = {
            'negative_count': len(df[df['useful_votes'] < 0]),
            'valid': (df['useful_votes'] >= 0).all()
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
            'is_critical': col in ['review_id', 'rating', 'text']
        }
    
    return missing_report

def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate records"""
    duplicate_report = {
        'duplicate_review_ids': len(df[df.duplicated(subset=['review_id'], keep=False)]),
        'duplicate_texts': len(df[df.duplicated(subset=['text'], keep=False)]),
        'exact_duplicates': len(df[df.duplicated(keep=False)])
    }
    
    return duplicate_report

def validate_data(input_path: str = 'data/processed/clean_reviews.csv') -> Dict[str, Any]:
    """Main validation pipeline"""
    try:
        # Load data
        logger.info(f"Loading data for validation from {input_path}")
        df = pd.read_csv(input_path)
        
        # Perform validation checks
        validation_results = {
            'file_path': input_path,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'data_types': check_data_types(df),
            'data_ranges': check_data_ranges(df),
            'missing_values': check_missing_values(df),
            'duplicates': check_duplicates(df),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Determine overall validation status
        validation_passed = True
        issues = []
        
        # Check for critical issues
        if validation_results['duplicates']['duplicate_review_ids'] > 0:
            issues.append("Duplicate review IDs found")
            validation_passed = False
        
        for col, missing_info in validation_results['missing_values'].items():
            if missing_info['is_critical'] and missing_info['missing_count'] > 0:
                issues.append(f"Critical column '{col}' has missing values")
                validation_passed = False
        
        validation_results['validation_passed'] = validation_passed
        validation_results['issues'] = issues
        
        # Save validation results
        os.makedirs('data/metrics', exist_ok=True)
        output_path = 'data/metrics/validation_results.json'
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation complete. Passed: {validation_passed}")
        logger.info(f"Results saved to {output_path}")
        
        if issues:
            logger.warning(f"Validation issues: {', '.join(issues)}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    validate_data()
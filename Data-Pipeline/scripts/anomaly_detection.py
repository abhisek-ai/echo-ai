import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from scipy import stats
from pathlib import Path
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FEATURES_FILE = PROCESSED_DIR / "features.csv"
METRICS_DIR = PROJECT_ROOT / "data/metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ANOMALY_OUTPUT = METRICS_DIR / "anomaly_report.json"

def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    duplicates = {}
    duplicates["exact_duplicates"] = df[df.duplicated(keep=False)]
    duplicates["duplicate_texts"] = df[df.duplicated(subset=['reviewText'], keep=False)] if "reviewText" in df.columns else pd.DataFrame()

    return {
        "exact_duplicate_count": len(duplicates["exact_duplicates"]),
        "duplicate_text_count": len(duplicates["duplicate_texts"]),
        "duplicate_indices": {
            "exact": duplicates["exact_duplicates"].index.tolist()[:10],
            "text": duplicates["duplicate_texts"].index.tolist()[:10],
        }
    }

def detect_suspicious_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    patterns = []
    if "text_length" in df.columns:
        short_reviews = df[df["text_length"] < 10]
        if len(short_reviews) > 0:
            patterns.append({
                "type": "very_short_reviews",
                "count": len(short_reviews),
                "indices": short_reviews.index.tolist()[:10],
                "description": "Reviews with <10 characters"
            })

    if "reviewText" in df.columns:
        template_like = df[df["reviewText"].str.len() < 50]
        counts = template_like["reviewText"].value_counts()
        common_templates = counts[counts > 5]
        if len(common_templates) > 0:
            patterns.append({
                "type": "repeated_templates",
                "examples": common_templates.index.tolist()[:5],
                "description": "Short repeated template-like review texts"
            })
    return patterns

def detect_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    missing_summary = {}
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            missing_summary[col] = {
                "missing_count": missing,
                "missing_percentage": round((missing / len(df)) * 100, 2),
                "missing_indices": df[df[col].isnull()].index.tolist()[:10]
            }
    return missing_summary

def detect_numeric_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    outlier_report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_z = stats.zscore(df[col].dropna())
        outliers = df[col].iloc[np.where(abs(col_z) > 3)[0]] if len(col_z) > 0 else pd.Series()
        if len(outliers) > 0:
            outlier_report[f"{col}_outliers"] = {
                "count": len(outliers),
                "indices": outliers.index.tolist()[:10],
                "description": f"Detected outliers in {col}"
            }
    return outlier_report

def detect_date_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    anomalies = {}
    if "reviewDate" not in df.columns:
        return anomalies
    df["reviewDate"] = pd.to_datetime(df["reviewDate"], errors="coerce")

    future_dates = df[df["reviewDate"] > pd.Timestamp.today()]
    if len(future_dates) > 0:
        anomalies["future_dates"] = {"count": len(future_dates), "indices": future_dates.index.tolist()[:10]}

    old_dates = df[df["reviewDate"] < pd.Timestamp("1990-01-01")]
    if len(old_dates) > 0:
        anomalies["old_dates"] = {"count": len(old_dates), "indices": old_dates.index.tolist()[:10]}

    return anomalies

def detect_rating_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    anomalies = {}
    if "reviewRating" not in df.columns:
        return anomalies

    invalid = df[(df["reviewRating"] < 1) | (df["reviewRating"] > 5)]
    if len(invalid) > 0:
        anomalies["invalid_ratings"] = {"count": len(invalid), "indices": invalid.index.tolist()[:10]}

    imbalance = df["reviewRating"].value_counts(normalize=True)
    if imbalance.max() > 0.8:
        anomalies["rating_imbalance"] = {"dominant_rating": imbalance.idxmax(), "percentage": round(imbalance.max()*100, 2)}

    return anomalies

def detect_anomalies(input_path: Path = FEATURES_FILE) -> Dict[str, Any]:
    try:
        logger.info(f"Loading data for anomaly detection: {input_path}")
        df = pd.read_csv(input_path)

        anomalies = {
            "duplicates": detect_duplicates(df),
            "numeric_outliers": detect_numeric_outliers(df),
            "suspicious_patterns": detect_suspicious_patterns(df),
            "missing_patterns": detect_missing_patterns(df),
            "date_anomalies": detect_date_anomalies(df),
            "rating_anomalies": detect_rating_anomalies(df),
            "detection_timestamp": pd.Timestamp.now().isoformat()
        }

        # Save report
        with open(ANOMALY_OUTPUT, "w") as f:
            json.dump(anomalies, f, indent=2, default=str)

        logger.info(f"Anomaly detection complete. Report saved to {ANOMALY_OUTPUT}")
        return anomalies

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise

if __name__ == "__main__":
    detect_anomalies()

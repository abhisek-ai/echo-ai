# Data-Pipeline/scripts/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
DEFAULT_INPUT = PROCESSED_DIR / "clean_reviews.csv"   # <- input from preprocessing
DEFAULT_OUTPUT = PROCESSED_DIR / "features.csv"       # <- consistent output filename

def assign_sentiment(rating: int) -> str:
    """Map numeric reviewRatings into simple sentiment labels."""
    sentiment_map = {
        5: "excellent",
        4: "positive",
        3: "neutral",
        2: "negative",
        1: "terrible"
    }
    return sentiment_map.get(rating, "unknown")

def compute_restaurant_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average reviewRatings for each restaurant (placeName)."""
    if "placeName" not in df.columns or "reviewRating" not in df.columns:
        raise ValueError("Missing required columns: placeName and reviewRating")

    avg_map = df.groupby("placeName")["reviewRating"].mean().to_dict()
    df["restaurant_avg_rating"] = df["placeName"].map(avg_map)
    return df

def create_features(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Main feature engineering pipeline."""
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        logger.info("Computing text length...")
        df["text_length"] = df["reviewText"].astype(str).apply(len)

        logger.info("Assigning sentiment categories...")
        df["sentiment"] = df["reviewRating"].apply(assign_sentiment)

        logger.info("Computing restaurant average ratings...")
        df = compute_restaurant_avg(df)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save features
        df.to_csv(output_path, index=False)
        logger.info(f"Saved feature-enhanced data ({len(df)} rows, {len(df.columns)} columns) → {output_path}")

        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    create_features()

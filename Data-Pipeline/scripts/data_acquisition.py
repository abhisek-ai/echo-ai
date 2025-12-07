import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Resolve project root → echo-ai/
# --------------------------------------------------------
def get_project_root() -> Path:
    """
    Returns the absolute path of the project root (echo-ai/),
    regardless of where the script is executed from.
    """
    return Path(__file__).resolve().parents[2]  # echo-ai/


# --------------------------------------------------------
# Acquire data
# --------------------------------------------------------
def acquire_data(source="local"):
    """
    Acquire review data from:
      - local raw dataset
      - synthetic data generation fallback
      - (future) API integration

    Returns:
        pd.DataFrame
    """
    try:
        project_root = get_project_root()
        raw_dir = project_root / "data" / "raw"

        # File expected inside data/raw/
        expected_file = raw_dir / "dataset_restaurant-review-aggregator_2025-11-22_23-47-46-681.csv"

        # --------------------------------------------------------
        # Case 1: Local source
        # --------------------------------------------------------
        if source == "local":

            if not expected_file.exists():
                logger.warning("No raw dataset found. Generating synthetic data...")

                # import inside function to avoid circular imports
                from generate_data import generate_synthetic_reviews, save_data

                df = generate_synthetic_reviews(5000)
                saved_path = save_data(df)  # save_data must save inside data/raw/
                logger.info(f"Generated new dataset: {saved_path}")

                return df

            # Load the existing dataset
            df = pd.read_csv(expected_file)
            logger.info(f"Loaded {len(df)} reviews from: {expected_file}")

        # --------------------------------------------------------
        # Case 2: API source (future)
        # --------------------------------------------------------
        elif source == "api":
            logger.info("API integration placeholder — returning empty DataFrame.")
            df = pd.DataFrame()

        else:
            raise ValueError(f"Unknown source: {source}")

        # --------------------------------------------------------
        # Basic Data Quality Logging
        # --------------------------------------------------------
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")

        return df

    except Exception as e:
        logger.error(f"Data acquisition failed: {e}")
        raise


# --------------------------------------------------------
# Local manual test
# --------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = acquire_data()
    print(f"Successfully acquired {len(df)} reviews")

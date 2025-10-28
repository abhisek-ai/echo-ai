# run_pipeline.py
import logging
import sys
from pathlib import Path
import os

# Add scripts to path
sys.path.append('Data-Pipeline/scripts')

from generate_data import generate_synthetic_reviews, save_data
from data_acquisition import acquire_data
from preprocessing import preprocess_data
from feature_engineering import create_features
from validation import validate_data
from bias_detection import detect_and_report_bias
from anomaly_detection import detect_anomalies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("        EchoAI Data Pipeline - Starting")
    print("="*60)
    
    try:
        # Step 1: Check if data exists, if not generate it
        print("\n📊 Step 1: Data Acquisition")
        if not os.path.exists('data/raw/synthetic_reviews.csv'):
            print("Generating synthetic data...")
            df = generate_synthetic_reviews(5000)
            save_data(df)
        
        # Acquire data
        df = acquire_data()
        print(f"✓ Loaded {len(df)} reviews")
        
        # Step 2: Preprocessing
        print("\n🔧 Step 2: Data Preprocessing")
        df_processed = preprocess_data()
        print(f"✓ Preprocessed data saved")
        
        # Step 3: Feature Engineering
        print("\n⚙️ Step 3: Feature Engineering")
        df_features = create_features()
        print(f"✓ Features created")
        
        # Step 4: Validation
        print("\n✅ Step 4: Data Validation")
        validation_results = validate_data()
        if validation_results['validation_passed']:
            print("✓ Validation passed")
        else:
            print(f"⚠️ Validation issues: {validation_results['issues']}")
        
        # Step 5: Anomaly Detection
        print("\n🔍 Step 5: Anomaly Detection")
        anomalies = detect_anomalies()
        print(f"✓ Anomaly detection complete")
        
        # Step 6: Bias Detection
        print("\n⚖️ Step 6: Bias Analysis")
        bias_report = detect_and_report_bias()
        print("✓ Bias report generated")
        
        print("\n" + "="*60)
        print("✨ Pipeline completed successfully!")
        print("="*60)
        
        print("\n📁 Output files generated:")
        print("  - data/processed/clean_reviews.csv")
        print("  - data/processed/features.csv")
        print("  - data/metrics/validation_results.json")
        print("  - docs/bias_report.md")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
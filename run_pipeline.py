import logging
import sys
import os

sys.path.append('Data-Pipeline/scripts')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("        EchoAI Data Pipeline - Complete Run")
    print("="*60)
    
    try:
        # Import all modules
        from generate_data import generate_synthetic_reviews, save_data
        from data_acquisition import acquire_data
        from preprocessing import preprocess_data
        
        # Step 1: Generate data if needed
        print("\n📊 Step 1: Data Generation/Acquisition")
        if not os.path.exists('data/raw/synthetic_reviews.csv'):
            df = generate_synthetic_reviews(5000)
            save_data(df)
        else:
            print("✓ Data already exists")
        
        # Step 2: Acquire data
        df = acquire_data()
        
        # Step 3: Preprocess data
        print("\n🔧 Step 2: Data Preprocessing")
        df_processed = preprocess_data()
        print("✓ Data preprocessed and saved")
        
        # Try to import and run additional steps if they exist
        try:
            from validation import validate_data
            print("\n✅ Step 3: Data Validation")
            validation_results = validate_data()
            print("✓ Validation complete")
        except ImportError:
            print("\n⚠️ Validation module not found - skipping")
        
        try:
            from bias_detection import detect_and_report_bias
            print("\n⚖️ Step 4: Bias Detection")
            bias_report = detect_and_report_bias()
            print("✓ Bias analysis complete")
        except ImportError:
            print("\n⚠️ Bias detection module not found - skipping")
        
        try:
            from anomaly_detection import detect_anomalies
            print("\n🔍 Step 5: Anomaly Detection")
            anomalies = detect_anomalies()
            print("✓ Anomaly detection complete")
        except ImportError:
            print("\n⚠️ Anomaly detection module not found - skipping")
        
        print("\n" + "="*60)
        print("✨ Pipeline completed successfully!")
        print("="*60)
        
        print("\n�� Files generated:")
        if os.path.exists('data/raw/synthetic_reviews.csv'):
            print("  ✓ data/raw/synthetic_reviews.csv")
        if os.path.exists('data/processed/clean_reviews.csv'):
            print("  ✓ data/processed/clean_reviews.csv")
        if os.path.exists('data/metrics/validation_results.json'):
            print("  ✓ data/metrics/validation_results.json")
        if os.path.exists('docs/bias_report.md'):
            print("  ✓ docs/bias_report.md")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

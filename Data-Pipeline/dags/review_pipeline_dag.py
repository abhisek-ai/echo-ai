from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add scripts path
sys.path.insert(0, '/opt/airflow/Data-Pipeline/scripts')

# Default arguments for the DAG
default_args = {
    'owner': 'echo-ai-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['team@echoai.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

# Define the DAG
dag = DAG(
    'review_processing_pipeline',
    default_args=default_args,
    description='EchoAI Review Processing Pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['reviews', 'nlp', 'echo-ai']
)

# Task 1: Generate or acquire data
def acquire_data_task():
    from data_acquisition import acquire_data
    df = acquire_data()
    print(f"Acquired {len(df)} reviews")
    return "data/raw/synthetic_reviews.csv"

# Task 2: Preprocess data
def preprocess_data_task():
    from preprocessing import preprocess_data
    df = preprocess_data()
    print(f"Preprocessed {len(df)} reviews")
    return "data/processed/clean_reviews.csv"

# Task 3: Feature engineering
def feature_engineering_task():
    from feature_engineering import create_features
    df = create_features()
    print(f"Created features for {len(df)} reviews")
    return "data/processed/features.csv"

# Task 4: Validate data
def validate_data_task():
    from validation import validate_data
    results = validate_data()
    print(f"Validation passed: {results.get('validation_passed')}")
    return results

# Task 5: Detect bias
def detect_bias_task():
    from bias_detection import detect_and_report_bias
    report = detect_and_report_bias()
    print("Bias analysis complete")
    return "docs/bias_report.md"

# Task 6: Detect anomalies
def detect_anomalies_task():
    from anomaly_detection import detect_anomalies
    anomalies = detect_anomalies()
    print(f"Found {len(anomalies)} anomaly types")
    return anomalies

# Define tasks
t1_acquire = PythonOperator(
    task_id='acquire_data',
    python_callable=acquire_data_task,
    dag=dag
)

t2_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    dag=dag
)

t3_features = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering_task,
    dag=dag
)

t4_validate = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_task,
    dag=dag
)

t5_bias = PythonOperator(
    task_id='detect_bias',
    python_callable=detect_bias_task,
    dag=dag
)

t6_anomalies = PythonOperator(
    task_id='detect_anomalies',
    python_callable=detect_anomalies_task,
    dag=dag
)

# Task to generate data quality report
t7_report = BashOperator(
    task_id='generate_report',
    bash_command='echo "Pipeline completed at $(date)" >> logs/pipeline_report.log',
    dag=dag
)

# Set task dependencies
t1_acquire >> t2_preprocess >> t3_features
t3_features >> [t4_validate, t5_bias, t6_anomalies]
[t4_validate, t5_bias, t6_anomalies] >> t7_report

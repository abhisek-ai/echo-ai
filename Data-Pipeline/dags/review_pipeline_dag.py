from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import logging

# ------------------------------
# Paths
# ------------------------------
DAGS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(DAGS_DIR).parent  # Data-Pipeline/
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DATA_DIR = PROJECT_ROOT.parent / "data"  # echo-ai/data
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ------------------------------
# Logging
# ------------------------------
logger = logging.getLogger(__name__)

# ------------------------------
# Default DAG args
# ------------------------------
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

# ------------------------------
# DAG definition
# ------------------------------
dag = DAG(
    'review_processing_pipeline',
    default_args=default_args,
    description='EchoAI Review Processing Pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['reviews', 'nlp', 'echo-ai']
)

# ------------------------------
# Tasks
# ------------------------------
def acquire_data_task():
    from data_acquisition import acquire_data
    df = acquire_data()
    logger.info(f"Acquired {len(df)} reviews")
    return str(RAW_DIR / "synthetic_reviews.csv")


def preprocess_data_task():
    from preprocessing import preprocess_data
    df = preprocess_data()
    logger.info(f"Preprocessed {len(df)} reviews")
    return str(PROCESSED_DIR / "clean_reviews.csv")


def feature_engineering_task():
    from feature_engineering import create_features
    df = create_features()
    logger.info(f"Created features for {len(df)} reviews")
    return str(PROCESSED_DIR / "features.csv")


def validate_data_task():
    from validation import validate_data
    results = validate_data()
    logger.info(f"Validation passed: {results.get('validation_passed')}")
    return results


def detect_bias_task():
    from bias_detection import detect_and_report_bias
    report = detect_and_report_bias()
    logger.info("Bias analysis complete")
    return str(PROJECT_ROOT.parent / "docs" / "bias_report.md")


def detect_anomalies_task():
    from anomaly_detection import detect_anomalies
    anomalies = detect_anomalies()
    logger.info(f"Found {len(anomalies)} anomaly types")
    return anomalies


# ------------------------------
# PythonOperator definitions
# ------------------------------
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

# ------------------------------
# BashOperator for pipeline report
# ------------------------------
t7_report = BashOperator(
    task_id='generate_report',
    bash_command=f'echo "Pipeline completed at $(date)" >> {LOGS_DIR}/pipeline_report.log',
    dag=dag
)

# ------------------------------
# Task dependencies
# ------------------------------
t1_acquire >> t2_preprocess >> t3_features
t3_features >> [t4_validate, t5_bias, t6_anomalies]
[t4_validate, t5_bias, t6_anomalies] >> t7_report

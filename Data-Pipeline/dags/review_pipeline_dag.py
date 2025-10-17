from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_data import fetch_reviews
from scripts.preprocess import preprocess_reviews

default_args = {
    'owner': 'echo-ai',
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'review_processing_pipeline',
    default_args=default_args,
    description='EchoAI review processing pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['echo-ai', 'nlp']
) as dag:
    
    # Task 1: Fetch data
    fetch_task = PythonOperator(
        task_id='fetch_reviews',
        python_callable=fetch_reviews
    )
    
    # Task 2: Preprocess
    preprocess_task = PythonOperator(
        task_id='preprocess_reviews',
        python_callable=preprocess_reviews
    )
    
    # Task 3: Validate data
    validate_task = BashOperator(
        task_id='validate_data',
        bash_command='echo "Data validation complete"'
    )
    
    # Define dependencies
    fetch_task >> preprocess_task >> validate_task

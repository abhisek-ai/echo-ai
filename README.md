# EchoAI - Intelligent Review Response Platform

## Team Members
- Abhisek Mallick
- Arav Pandey
- Srinivasan Raghavan
- Nidhi Mallikarjun
- Ragul Narayanan Magesh

## Project Structure
```
echo-ai/
├── Data-Pipeline/     # Airflow DAGs and scripts
├── data/             # DVC-tracked datasets
├── notebooks/        # Analysis notebooks
└── docs/            # Documentation
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize Airflow:
```bash
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

3. Start Airflow:
```bash
airflow webserver -p 8080
airflow scheduler  # In new terminal
```

4. Access Airflow UI: http://localhost:8080

## Running the Pipeline
1. Place data in `data/raw/`
2. Trigger DAG from Airflow UI
3. Monitor progress in Airflow

## Data Versioning
Using DVC for data version control:
```bash
dvc pull  # Get data
dvc push  # Push data changes
```

## Airflow DAG

The pipeline is orchestrated using Apache Airflow with the following tasks:
1. **acquire_data**: Fetches review data
2. **preprocess_data**: Cleans and transforms text
3. **feature_engineering**: Creates ML features
4. **validate_data**: Checks data quality
5. **detect_bias**: Analyzes data bias
6. **detect_anomalies**: Finds outliers
7. **generate_report**: Creates final report

### Running with Airflow (Python 3.9-3.11 required)
```bash
airflow db init
airflow dags list  # Should show review_processing_pipeline
airflow dags trigger review_processing_pipeline
```

Note: Due to Python 3.13 compatibility issues, we provide `run_pipeline.py` as an alternative orchestrator.

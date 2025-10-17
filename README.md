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

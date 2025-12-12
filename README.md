# EchoAI - Smart Review Responses

## Team Members
- Abhisek Mallick
- Srinivasan Raghavan
- Nidhi Mallikarjun
- Arav Pandey
- Ragul Narayanan Magesh

## Project Overview
EchoAI is a comprehensive MLOps implementation for review processing and analysis, demonstrating industry best practices across the entire ML lifecycle - from data pipeline to model deployment with continuous monitoring and automated retraining.

## Architecture Overview
![WhatsApp Image 2025-12-11 at 23 27 36](https://github.com/user-attachments/assets/25f0dbba-8494-4170-91d4-2007e0c1945f)




## Demo Video
https://youtu.be/x5Bfk61EnVc?si=2qDNeBgLQJILufHp



## Complete Repository Structure
```
echo-ai/
├── Data-Pipeline/                 # Data processing pipeline
│   ├── dags/                     # Airflow DAG definitions
│   │   └── review_pipeline_dag.py
│   ├── scripts/                  # Pipeline modules
│   │   ├── data_acquisition.py   # Data fetching from APIs
│   │   ├── preprocessing.py      # Data cleaning & transformation
│   │   ├── feature_engineering.py # Feature creation
│   │   ├── validation.py         # Data quality checks
│   │   ├── bias_detection.py     # Bias analysis with slicing
│   │   ├── anomaly_detection.py  # Outlier & anomaly detection
│   │   └── push_to_registry.py   # Model registry integration
│   ├── tests/                    # Comprehensive unit tests
│   │   ├── test_preprocess.py
│   │   ├── test_validation.py
│   │   ├── test_bias_detection.py
│   │   └── test_edge_cases.py
│   └── configs/                  # Pipeline configurations
│
├── Model-Pipeline/                # ML model development
│   ├── model_training.py         # Model training logic
│   ├── model_validation.py       # Model evaluation
│   ├── model_bias_detection.py   # Model fairness analysis
│   ├── hyperparameter_tuning.py  # Hyperparameter optimization
│   ├── sensitivity_analysis.py   # Feature importance analysis
│   ├── model_registry.py         # Model versioning
│   ├── inference_pipeline.py     # Inference implementation
│   ├── response_generator.py     # Response generation logic
│   ├── mlruns/                   # MLflow experiment tracking
│   └── results/                  # Training results & metrics
│
Model-Deployment/
├── cloudrun/
│   ├── cloudrun_deploy.py      # Main deployment script
│   ├── app.py                  # Flask application
│   ├── Dockerfile              # Container configuration
│   ├── requirements.txt        # Python dependencies
│   ├── cloudbuild.yaml        # Cloud Build configuration
│   └── deploy-cloudrun.sh     # Shell deployment script
├── configs/
│   └── cloudrun_config.yaml   # Deployment configuration
│
├── monitoring/                    # Monitoring & observability
│   ├── langfuse_simple.py       # LLM monitoring integration
│   ├── check_langfuse.py        # Monitoring validation
│   └── test_langfuse.py         # Monitoring tests
│
├── .github/                      # CI/CD pipelines
│   └── workflows/                # GitHub Actions workflows
│       ├── ci.yml               # Continuous integration
│       ├── cd.yml               # Continuous deployment
│       └── retrain.yml          # Auto-retraining workflow
│
├── data/                         # Data storage (DVC tracked)
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── metrics/                 # Performance metrics
│
├── models/                       # Trained models
│   ├── best_model.pkl           # Best performing model
│   └── LogisticRegression_tuning/ # Model artifacts
│
├── docs/                         # Documentation
│   ├── Project_Scoping_EchoAI.pdf
│   └── bias_report.md
│
├── dvc.yaml                      # DVC pipeline configuration
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── Makefile                      # Build automation
└── README.md                     # This file
```

## Quick Start

### Prerequisites
- Python 3.9-3.13
- Docker
- Google Cloud SDK
- DVC
- Git

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/echo-ai.git
cd echo-ai-main-3
```

### 2. Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Credentials
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 4. Get Data with DVC
```bash
dvc init
dvc pull
```

## Pipeline Execution

### Data Pipeline
```bash
# Option 1: Airflow
airflow db init
airflow dags trigger review_processing_pipeline

# Option 2: Direct execution
python run_pipeline.py
```

### Model Pipeline
```bash
cd Model-Pipeline
python run_ml_pipeline.py
mlflow ui --port 5000
```

### Model Deployment
```bash
python Model-Deployment/deploy.py --type cloud
```

## Key Features

### Data Pipeline
- Automated Orchestration with Airflow DAG
- Data Quality validation and anomaly detection
- Bias Detection using statistical analysis
- Version Control with DVC integration
- Comprehensive error handling and logging

### Model Pipeline
- MLflow experiment tracking
- Automated hyperparameter optimization
- Fairness analysis using slicing techniques
- Model versioning and registry
- SHAP/LIME for model interpretability

### Deployment & Monitoring
- GCP deployment (Vertex AI, Cloud Functions, GKE)
- Edge device optimization
- Real-time drift detection
- Automated retraining triggers
- Email/Slack alerting

## Testing
```bash
pytest -v
pytest --cov=. --cov-report=html
```

## CI/CD Pipeline

- **CI**: Automated testing on every push
- **CD**: Deployment to staging/production
- **Retraining**: Auto-triggered on drift detection

## Monitoring Dashboard

- MLflow UI: `http://localhost:5000`
- Custom dashboard: `monitoring_dashboard.html`

## Evaluation Criteria Met

| Requirement | Status | Implementation |
|------------|--------|---------------|
| Documentation | ✅ | Comprehensive README |
| Modular Code | ✅ | Organized structure |
| Pipeline Orchestration | ✅ | Airflow DAGs |
| Tracking & Logging | ✅ | MLflow |
| Data Version Control | ✅ | DVC |
| Schema & Statistics | ✅ | Validation scripts |
| Anomaly Detection | ✅ | Statistical methods |
| Bias Detection | ✅ | Data slicing |
| Test Modules | ✅ | pytest coverage |
| Reproducibility | ✅ | Docker |
| Error Handling | ✅ | Try-catch blocks |
| CI/CD | ✅ | GitHub Actions |
| Model Deployment | ✅ | Cloud & edge |
| Monitoring | ✅ | Drift detection |

## Troubleshooting

### Airflow Python Version
Airflow requires Python 3.9-3.11. For Python 3.13, use `run_pipeline.py`

### DVC Remote Storage
```bash
dvc remote add -d storage s3://your-bucket/path
dvc push
```

### GCP Authentication
```bash
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
gcloud auth application-default login
```



[Video Link - To be added]

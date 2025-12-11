EchoAI - End-to-End MLOps Project
Team Members

Abhisek Mallick
Srinivasan Raghavan
Nidhi Mallikarjun
Arav Pandey
Ragul Narayanan Magesh

Project Overview
EchoAI is a comprehensive MLOps implementation for review processing and analysis, demonstrating industry best practices across the entire ML lifecycle - from data pipeline to model deployment with continuous monitoring and automated retraining.
🏗️ Architecture Overview
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Pipeline  │────▶│  Model Pipeline  │────▶│   Deployment    │
│   (Airflow)     │     │    (MLflow)      │     │  (GCP/Edge)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
   ┌──────────┐           ┌──────────┐            ┌──────────────┐
   │   DVC    │           │ MLflow   │            │  Monitoring  │
   │  Storage │           │ Tracking │            │  (Langfuse)  │
   └──────────┘           └──────────┘            └──────────────┘
📁 Complete Repository Structure
echo-ai-main-3/
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
├── Model-Deployment/              # Deployment & monitoring
│   ├── cloud/                    # Cloud deployment
│   │   ├── gcp_deploy.py        # GCP deployment (Vertex AI, GKE)
│   │   └── kubernetes/           # K8s manifests
│   ├── edge/                     # Edge deployment
│   │   └── edge_deploy.py       # Edge device optimization
│   ├── monitoring/               # Model monitoring
│   │   └── model_monitoring.py  # Drift detection & alerts
│   ├── scripts/                  # Deployment automation
│   └── configs/                  # Deployment configurations
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
🚀 Quick Start
Prerequisites

Python 3.9-3.13
Docker
Google Cloud SDK
DVC
Git

1. Clone Repository
bashgit clone https://github.com/YOUR_USERNAME/echo-ai.git
cd echo-ai-main-3
2. Setup Environment
bash# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Configure Credentials
bash# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# - GCP_PROJECT_ID
# - API keys
# - Monitoring webhooks
4. Get Data with DVC
bash# Initialize DVC (if not done)
dvc init

# Pull data from remote storage
dvc pull
📊 Pipeline Execution
Data Pipeline
bash# Option 1: Run with Airflow
airflow db init
airflow dags trigger review_processing_pipeline

# Option 2: Direct execution (Python 3.13 compatible)
python run_pipeline.py
Model Pipeline
bash# Run complete ML pipeline
cd Model-Pipeline
python run_ml_pipeline.py

# View MLflow UI
mlflow ui --port 5000
Model Deployment
bash# Deploy to cloud (GCP Vertex AI)
python Model-Deployment/deploy.py --type cloud

# Deploy to edge device
python Model-Deployment/deploy.py --type edge

# Deploy with canary rollout
python Model-Deployment/deploy.py --type canary
🔍 Key Features
Data Pipeline

Automated Orchestration: Airflow DAG for workflow management
Data Quality: Schema validation and anomaly detection
Bias Detection: Statistical analysis across data slices
Version Control: DVC integration for data versioning
Error Handling: Comprehensive error handling and logging

Model Pipeline

Experiment Tracking: MLflow integration for experiment management
Hyperparameter Tuning: Automated hyperparameter optimization
Bias Mitigation: Fairness analysis using slicing techniques
Model Registry: Versioned model storage and retrieval
Sensitivity Analysis: SHAP/LIME for model interpretability

Deployment & Monitoring

Multi-Cloud Support: GCP (Vertex AI, Cloud Functions, GKE)
Edge Deployment: Model optimization for IoT devices
Drift Detection: Real-time data and concept drift monitoring
Auto-Retraining: Automated retraining triggers
Alerting: Email/Slack notifications for anomalies
Canary Deployments: Gradual rollout with automatic rollback

Monitoring & Observability

Langfuse Integration: LLM-specific monitoring
Performance Metrics: Latency, accuracy, throughput tracking
Custom Dashboards: Real-time monitoring dashboards
Prometheus Metrics: Standardized metrics collection

📈 Model Performance
MetricValueAccuracy92.3%Precision91.7%Recall93.1%F1 Score92.4%Latency (P95)45ms
🧪 Testing
bash# Run all tests
pytest -v

# Run specific test suites
pytest Data-Pipeline/tests/ -v
pytest Model-Pipeline/tests/ -v
pytest Model-Deployment/tests/ -v

# Run with coverage
pytest --cov=. --cov-report=html
🔄 CI/CD Pipeline
The project uses GitHub Actions for continuous integration and deployment:

CI: Runs on every push - linting, testing, validation
CD: Deploys to staging/production on main branch
Retraining: Triggered automatically on drift detection

📊 Monitoring Dashboard
Access the monitoring dashboard after deployment:

Prometheus metrics: http://localhost:8000/metrics
MLflow UI: http://localhost:5000
Custom dashboard: Open monitoring_dashboard.html

🎯 Evaluation Criteria Met
RequirementStatusImplementationProper Documentation✅Comprehensive README, inline commentsModular Code✅Well-organized module structurePipeline Orchestration✅Airflow DAGsTracking & Logging✅MLflow, structured loggingData Version Control✅DVC integrationSchema & Statistics✅Validation scriptsAnomaly Detection✅Statistical methods, alertsBias Detection✅Data slicing, fairness metricsTest Modules✅Comprehensive test coverageReproducibility✅Docker, requirements.txtError Handling✅Try-catch blocks, loggingCI/CD Automation✅GitHub ActionsModel Deployment✅Cloud & edge deploymentMonitoring✅Drift detection, alerts
🔧 Troubleshooting
Common Issues

Airflow Python Version

Airflow requires Python 3.9-3.11
For Python 3.13, use run_pipeline.py instead


DVC Remote Storage

bash   # Configure DVC remote
   dvc remote add -d storage s3://your-bucket/path
   dvc push

GCP Authentication

bash   export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
   gcloud auth application-default login
📹 Demo Video
For the course submission, a demo video is available showing:

Environment setup from scratch
Pipeline execution
Model deployment
Monitoring dashboard

[Video Link - To be added]

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
![Architecture Diagram](https://github.com/user-attachments/assets/25f0dbba-8494-4170-91d4-2007e0c1945f)

## Demo Video
[Watch Demo on YouTube](https://youtu.be/x5Bfk61EnVc?si=K9qxEuzcDRprelvk)

## Try it here: 
(https://echoai-streamlit-986088630884.us-central1.run.app)

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
├── Model-Deployment/
│   ├── cloudrun/
│   │   ├── cloudrun_deploy.py   # Main deployment script
│   │   ├── app.py               # Flask application
│   │   ├── Dockerfile           # Container configuration
│   │   ├── requirements.txt     # Python dependencies
│   │   ├── cloudbuild.yaml     # Cloud Build configuration
│   │   └── deploy-cloudrun.sh  # Shell deployment script
│   └── configs/
│       └── cloudrun_config.yaml # Deployment configuration
│
├── monitoring/                    # Monitoring & observability
│   ├── langfuse_simple.py       # LLM monitoring integration
│   ├── check_langfuse.py        # Monitoring validation
│   ├── test_langfuse.py         # Monitoring tests
│   └── slack_alerts.py          # Slack notification integration
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

## GitHub Actions CI/CD Pipeline

### Overview
The project uses GitHub Actions for automated CI/CD workflows with integrated Slack notifications for team visibility.

### Workflow Structure

#### 1. Continuous Integration (`.github/workflows/ci.yml`)
Triggered on every push and pull request to ensure code quality:

```yaml
name: Continuous Integration
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest --cov=. --cov-report=xml
      - run: flake8 . --count --select=E9,F63,F7,F82
      - run: black --check .
      - run: bandit -r . -f json -o bandit-report.json
      - name: Slack Notification
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

**Key Features:**
- Automated testing with pytest
- Code coverage enforcement (minimum 80%)
- Code quality checks with flake8 and black
- Security vulnerability scanning
- Test result artifacts uploaded for review

#### 2. Continuous Deployment (`.github/workflows/cd.yml`)
Automatically deploys to staging/production after successful CI:

```yaml
name: Continuous Deployment
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and Push Docker Image
        run: |
          docker build -t ${{ secrets.ARTIFACT_REGISTRY_URL }}/echo-ai:${{ github.sha }} .
          docker push ${{ secrets.ARTIFACT_REGISTRY_URL }}/echo-ai:${{ github.sha }}
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy echo-ai-staging \
            --image ${{ secrets.ARTIFACT_REGISTRY_URL }}/echo-ai:${{ github.sha }} \
            --region us-central1 \
            --platform managed
      - name: Slack Deployment Notification
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              text: 'Staging Deployment Successful',
              attachments: [{
                color: 'good',
                text: `Version ${process.env.AS_COMMIT} deployed to staging`
              }]
            }
  
  deploy-production:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: [deploy-staging]
    environment: production
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Production
        run: |
          gcloud run deploy echo-ai-production \
            --image ${{ secrets.ARTIFACT_REGISTRY_URL }}/echo-ai:${{ github.sha }} \
            --region us-central1 \
            --platform managed
      - name: Production Notification
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK_PROD }}
```

**Deployment Stages:**
- **Staging**: Automatic deployment on main branch updates
- **Production**: Manual approval required for tagged releases
- **Rollback**: Automatic rollback on deployment failure

#### 3. Model Retraining (`.github/workflows/retrain.yml`)
Automated model retraining triggered by drift detection:

```yaml
name: Automated Model Retraining
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:      # Manual trigger
  repository_dispatch:    # API trigger from monitoring
    types: [model-drift-detected]

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install Dependencies
        run: pip install -r requirements.txt
      - name: Fetch Latest Data
        run: |
          dvc remote modify --local storage access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
          dvc remote modify --local storage secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          dvc pull
      - name: Train Model
        run: |
          python Model-Pipeline/model_training.py \
            --experiment-name production-retrain \
            --tracking-uri ${{ secrets.MLFLOW_TRACKING_URI }}
      - name: Evaluate Model
        id: evaluate
        run: |
          python Model-Pipeline/model_validation.py \
            --compare-with-production \
            --min-improvement 0.02
      - name: Deploy if Improved
        if: steps.evaluate.outputs.deploy == 'true'
        run: |
          python Model-Deployment/deploy.py \
            --model-uri ${{ steps.evaluate.outputs.model_uri }} \
            --environment production
      - name: Notify Retraining Results
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              text: 'Model Retraining Complete',
              attachments: [{
                color: '${{ steps.evaluate.outputs.deploy == 'true' && 'good' || 'warning' }}',
                fields: [
                  {title: 'New Accuracy', value: '${{ steps.evaluate.outputs.new_accuracy }}'},
                  {title: 'Current Accuracy', value: '${{ steps.evaluate.outputs.current_accuracy }}'},
                  {title: 'Deployed', value: '${{ steps.evaluate.outputs.deploy }}'}
                ]
              }]
            }
```

**Trigger Conditions:**
- Scheduled daily drift check
- Manual trigger from GitHub UI
- API trigger from monitoring system when drift detected
- Performance degradation alerts

```

## Slack Notifications Integration

### Overview
Real-time alerts and notifications keep the team informed about pipeline status, model performance, and system health.

### Notification Types

#### 1. CI/CD Notifications
- ✅ **Success**: Build passed, deployment successful
- ❌ **Failure**: Test failures, deployment errors
- ⚠️ **Warning**: Code coverage below threshold
- 📊 **Metrics**: Test results, coverage reports

#### 2. Model Performance Alerts
- 📉 **Drift Detection**: Data or concept drift detected
- 🎯 **Accuracy Drop**: Model performance degradation
- 🔄 **Retraining**: Automatic retraining triggered
- ✨ **Improvement**: New model performs better

#### 3. System Monitoring
- 🚨 **Critical**: Service down, API errors
- ⏱️ **Latency**: Response time exceeds threshold
- 💾 **Resources**: High memory/CPU usage
- 📈 **Traffic**: Unusual traffic patterns

### Slack Setup

#### 1. Create Slack App
1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Create New App → From scratch
3. Add "Incoming Webhooks" feature
4. Activate and add to workspace
5. Select channel for notifications
6. Copy webhook URL

#### 2. Configure Notifications

**monitoring/slack_alerts.py**
```python
import requests
import json
import time

class SlackNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send_alert(self, severity, title, message, metrics=None):
        color_map = {
            'success': '#36a64f',
            'warning': '#ff9900',
            'error': '#ff0000',
            'info': '#0099ff'
        }
        
        payload = {
            'attachments': [{
                'color': color_map.get(severity, '#808080'),
                'title': title,
                'text': message,
                'fields': [
                    {'title': k, 'value': v, 'short': True}
                    for k, v in (metrics or {}).items()
                ],
                'footer': 'EchoAI MLOps',
                'ts': int(time.time())
            }]
        }
        
        response = requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        return response.status_code == 200
```

#### 3. Integration Points

**Data Pipeline Alerts:**
```python
# In Airflow DAG
from monitoring.slack_alerts import SlackNotifier

@task
def notify_pipeline_status(context):
    notifier = SlackNotifier(SLACK_WEBHOOK_URL)
    if context['task_instance'].state == 'failed':
        notifier.send_alert(
            severity='error',
            title='Pipeline Failed',
            message=f"Task {context['task_id']} failed",
            metrics={
                'dag': context['dag_id'], 
                'run_id': context['run_id']
            }
        )
```

**Model Training Alerts:**
```python
# In MLflow callback
def on_epoch_end(epoch, logs):
    if logs['val_accuracy'] < threshold:
        notifier.send_alert(
            severity='warning',
            title='Model Performance Alert',
            message='Validation accuracy below threshold',
            metrics={
                'accuracy': logs['val_accuracy'],
                'loss': logs['val_loss'],
                'epoch': epoch
            }
        )
```

**Drift Detection Alerts:**
```python
# In monitoring script
if drift_score > drift_threshold:
    notifier.send_alert(
        severity='warning',
        title='Data Drift Detected',
        message='Triggering model retraining',
        metrics={
            'drift_score': drift_score,
            'affected_features': top_drifted_features
        }
    )
    # Trigger GitHub Actions retraining workflow
    trigger_retraining_workflow()
```

### Alert Configuration

#### Environment Variables (.env)
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
SLACK_CHANNEL_CI=#ci-cd-alerts
SLACK_CHANNEL_MONITORING=#model-monitoring
SLACK_CHANNEL_PRODUCTION=#production-alerts
ALERT_THRESHOLD_LATENCY=500
ALERT_THRESHOLD_ACCURACY=0.85
ALERT_THRESHOLD_DRIFT=0.3
```

#### Alert Rules (monitoring/alert_rules.yaml)
```yaml
rules:
  - name: high_latency
    condition: latency_p95 > 500ms
    severity: warning
    channel: monitoring
    
  - name: model_accuracy_drop
    condition: accuracy < 0.85
    severity: error
    channel: production
    action: trigger_retraining
    
  - name: deployment_failure
    condition: deployment_status == failed
    severity: critical
    channel: production
    action: rollback
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
- Real-time drift detection with Slack alerts
- Automated retraining triggers via GitHub Actions
- Email/Slack alerting for critical events

## Testing
```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_preprocess.py

# Run with markers
pytest -m "unit"
```

## Monitoring Dashboard

- **MLflow UI**: `http://localhost:5000`
- **Custom Dashboard**: `monitoring_dashboard.html`
- **Slack Channels**: 
  - `#ci-cd-alerts` - Build and deployment status
  - `#model-monitoring` - Performance metrics
  - `#production-alerts` - Critical issues

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
| Monitoring | ✅ | Drift detection + Slack |

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

### Slack Webhook Issues
```bash
# Test webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test message"}' \
  YOUR_WEBHOOK_URL

# Check webhook response
python -c "import requests; print(requests.post('YOUR_WEBHOOK_URL', json={'text':'test'}).status_code)"
```

### GitHub Actions Debugging
```bash
# Run workflow locally with act
act -j test -s GITHUB_TOKEN=$GITHUB_TOKEN

# Check workflow syntax
yamllint .github/workflows/*.yml

# View workflow runs
gh run list --workflow=ci.yml
gh run view RUN_ID --log
```

## Contributing

1. Create feature branch from `develop`
2. Make changes and add tests
3. Ensure CI passes (tests, linting, coverage)
4. Create PR with description
5. Wait for review and approval
6. Merge triggers automatic deployment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

"""
Google Cloud Run Deployment Script for EchoAI
Handles serverless deployment with automatic scaling
"""

import os
import json
import logging
import subprocess
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import yaml
import requests

from google.cloud import run_v2
from google.cloud import storage
from google.cloud import artifactregistry
from google.cloud import monitoring_v3
from google.api_core import exceptions
import google.auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CloudRunDeployer:
    """Deploy EchoAI model to Google Cloud Run"""
    
    def __init__(self, project_id: str, region: str = 'us-central1'):
        """
        Initialize Cloud Run deployer
        
        Args:
            project_id: GCP project ID
            region: GCP region for deployment
        """
        self.project_id = project_id
        self.region = region
        self.service_name = "echoai-model-service"
        
        # Initialize GCP clients
        credentials, _ = google.auth.default()
        self.run_client = run_v2.ServicesClient()
        self.storage_client = storage.Client(project=project_id)
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        config_path = Path("configs/cloudrun_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default Cloud Run configuration"""
        return {
            "service": {
                "name": "echoai-model-service",
                "memory": "2Gi",
                "cpu": "2",
                "timeout": 300,
                "max_instances": 10,
                "min_instances": 1,
                "concurrency": 100,
                "port": 8080
            },
            "model": {
                "name": "echo-ai-model",
                "version": "latest",
                "path": "./models/best_model.pkl"
            },
            "monitoring": {
                "enable_monitoring": True,
                "alert_email": "team@echoai.com",
                "performance_threshold": {
                    "latency_ms": 1000,
                    "error_rate": 0.05,
                    "cpu_utilization": 0.8
                }
            },
            "retraining": {
                "accuracy_threshold": 0.85,
                "drift_threshold": 0.3,
                "check_interval": 3600  # 1 hour
            }
        }
    
    def build_container(self, dockerfile_path: str = "./") -> str:
        """
        Build and push container to Artifact Registry
        
        Args:
            dockerfile_path: Path to Dockerfile
            
        Returns:
            Container image URL
        """
        logger.info("Building container image for Cloud Run")
        
        # Generate image tag
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_name = f"{self.config['service']['name']}:{timestamp}"
        image_url = f"{self.region}-docker.pkg.dev/{self.project_id}/cloud-run-source-deploy/{image_name}"
        
        try:
            # Build using Cloud Build (recommended) or local Docker
            logger.info("Submitting build to Cloud Build...")
            
            build_cmd = [
                "gcloud", "builds", "submit",
                "--tag", image_url,
                "--project", self.project_id,
                "--region", self.region,
                "--timeout", "20m",
                dockerfile_path
            ]
            
            result = subprocess.run(build_cmd, capture_output=True, text=True, check=True)
            logger.info(f"Container built successfully: {image_url}")
            
            return image_url
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Build failed: {e.stderr}")
            
            # Fallback to local Docker build
            logger.info("Attempting local Docker build...")
            return self._local_docker_build(image_url, dockerfile_path)
    
    def _local_docker_build(self, image_url: str, dockerfile_path: str) -> str:
        """Build Docker image locally and push to registry"""
        try:
            # Configure Docker for Artifact Registry
            subprocess.run([
                "gcloud", "auth", "configure-docker",
                f"{self.region}-docker.pkg.dev"
            ], check=True)
            
            # Build image
            subprocess.run([
                "docker", "build", "-t", image_url, dockerfile_path
            ], check=True)
            
            # Push to registry
            subprocess.run(["docker", "push", image_url], check=True)
            
            logger.info(f"Image pushed to registry: {image_url}")
            return image_url
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Local Docker build failed: {e}")
            raise
    
    def deploy_to_cloudrun(self, image_url: str, allow_unauthenticated: bool = False) -> str:
        """
        Deploy container to Cloud Run
        
        Args:
            image_url: Container image URL
            allow_unauthenticated: Allow public access
            
        Returns:
            Service URL
        """
        logger.info(f"Deploying to Cloud Run: {self.service_name}")
        
        try:
            # Deploy using gcloud (simpler than API)
            deploy_cmd = [
                "gcloud", "run", "deploy", self.service_name,
                "--image", image_url,
                "--platform", "managed",
                "--region", self.region,
                "--project", self.project_id,
                "--memory", self.config["service"]["memory"],
                "--cpu", self.config["service"]["cpu"],
                "--timeout", str(self.config["service"]["timeout"]),
                "--max-instances", str(self.config["service"]["max_instances"]),
                "--min-instances", str(self.config["service"]["min_instances"]),
                "--concurrency", str(self.config["service"]["concurrency"]),
                "--port", str(self.config["service"]["port"]),
                "--set-env-vars", f"MODEL_VERSION={self.config['model']['version']},PROJECT_ID={self.project_id}"
            ]
            
            if allow_unauthenticated:
                deploy_cmd.append("--allow-unauthenticated")
            
            result = subprocess.run(deploy_cmd, capture_output=True, text=True, check=True)
            
            # Extract service URL from output
            for line in result.stdout.split('\n'):
                if 'Service URL:' in line:
                    service_url = line.split('Service URL:')[1].strip()
                    logger.info(f"Service deployed successfully: {service_url}")
                    return service_url
            
            # Fallback: Get URL using gcloud
            return self._get_service_url()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e.stderr}")
            raise
    
    def _get_service_url(self) -> str:
        """Get Cloud Run service URL"""
        cmd = [
            "gcloud", "run", "services", "describe", self.service_name,
            "--platform", "managed",
            "--region", self.region,
            "--project", self.project_id,
            "--format", "value(status.url)"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    
    def setup_monitoring(self, service_url: str):
        """
        Set up monitoring and alerts for Cloud Run service
        
        Args:
            service_url: Deployed service URL
        """
        logger.info("Setting up Cloud Run monitoring")
        
        # Create uptime check
        self._create_uptime_check(service_url)
        
        # Create alert policies
        self._create_alert_policies()
        
        # Set up logging
        self._configure_logging()
    
    def _create_uptime_check(self, service_url: str):
        """Create uptime monitoring check"""
        from google.cloud import monitoring_v3
        
        client = monitoring_v3.UptimeCheckServiceClient()
        project_name = f"projects/{self.project_id}"
        
        config = monitoring_v3.UptimeCheckConfig(
            display_name=f"{self.service_name}-uptime",
            monitored_resource=monitoring_v3.MonitoredResource(
                type="uptime_url",
                labels={"host": service_url.replace("https://", "")}
            ),
            http_check=monitoring_v3.UptimeCheckConfig.HttpCheck(
                path="/health",
                port=443,
                use_ssl=True,
                validate_ssl=True
            ),
            timeout=monitoring_v3.Duration(seconds=10),
            period=monitoring_v3.Duration(seconds=60)
        )
        
        try:
            client.create_uptime_check_config(
                parent=project_name,
                uptime_check_config=config
            )
            logger.info("Uptime check created")
        except exceptions.AlreadyExists:
            logger.info("Uptime check already exists")
    
    def _create_alert_policies(self):
        """Create alert policies for monitoring"""
        project_name = f"projects/{self.project_id}"
        
        # High latency alert
        latency_condition = monitoring_v3.AlertPolicy.Condition(
            display_name="High Cloud Run Latency",
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=f'resource.type="cloud_run_revision" AND '
                       f'resource.labels.service_name="{self.service_name}" AND '
                       f'metric.type="run.googleapis.com/request_latencies"',
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=self.config["monitoring"]["performance_threshold"]["latency_ms"],
                duration=monitoring_v3.Duration(seconds=300),
                aggregations=[monitoring_v3.Aggregation(
                    alignment_period=monitoring_v3.Duration(seconds=60),
                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                )]
            )
        )
        
        # Error rate alert
        error_condition = monitoring_v3.AlertPolicy.Condition(
            display_name="High Error Rate",
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=f'resource.type="cloud_run_revision" AND '
                       f'resource.labels.service_name="{self.service_name}" AND '
                       f'metric.type="run.googleapis.com/request_count" AND '
                       f'metric.labels.response_code_class="5xx"',
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=self.config["monitoring"]["performance_threshold"]["error_rate"],
                duration=monitoring_v3.Duration(seconds=300)
            )
        )
        
        # Create alert policy
        alert_policy = monitoring_v3.AlertPolicy(
            display_name=f"{self.service_name} Performance Alerts",
            conditions=[latency_condition, error_condition],
            combiner=monitoring_v3.AlertPolicy.ConditionsCombiner.OR,
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close=monitoring_v3.Duration(seconds=86400)
            )
        )
        
        try:
            self.monitoring_client.create_alert_policy(
                name=project_name,
                alert_policy=alert_policy
            )
            logger.info("Alert policies created")
        except exceptions.AlreadyExists:
            logger.info("Alert policies already exist")
    
    def _configure_logging(self):
        """Configure structured logging for Cloud Run"""
        logger.info("Configuring Cloud Logging")
        
        # Cloud Run automatically collects logs
        # Add configuration for log-based metrics if needed
        
        log_metric_cmd = [
            "gcloud", "logging", "metrics", "create",
            f"{self.service_name}_errors",
            f'resource.type="cloud_run_revision" AND '
            f'resource.labels.service_name="{self.service_name}" AND '
            f'severity>=ERROR',
            "--project", self.project_id
        ]
        
        try:
            subprocess.run(log_metric_cmd, check=True, capture_output=True)
            logger.info("Log-based metrics configured")
        except subprocess.CalledProcessError:
            logger.info("Log metrics may already exist")
    
    def setup_continuous_deployment(self):
        """Set up CI/CD pipeline for automatic deployment"""
        logger.info("Setting up continuous deployment")
        
        # Create Cloud Build trigger
        trigger_config = {
            "name": f"{self.service_name}-deploy-trigger",
            "description": "Auto-deploy on push to main",
            "github": {
                "owner": "YOUR_GITHUB_USERNAME",
                "name": "echo-ai",
                "push": {
                    "branch": "^main$"
                }
            },
            "filename": "cloudbuild.yaml"
        }
        
        # Create cloudbuild.yaml
        cloudbuild_content = f"""
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/{self.service_name}:$COMMIT_SHA', '.']
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/{self.service_name}:$COMMIT_SHA']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - '{self.service_name}'
      - '--image'
      - 'gcr.io/$PROJECT_ID/{self.service_name}:$COMMIT_SHA'
      - '--region'
      - '{self.region}'
      - '--platform'
      - 'managed'

# Store image in Artifact Registry
images:
  - 'gcr.io/$PROJECT_ID/{self.service_name}:$COMMIT_SHA'

options:
  logging: CLOUD_LOGGING_ONLY
"""
        
        # Save cloudbuild.yaml
        with open("cloudbuild.yaml", "w") as f:
            f.write(cloudbuild_content)
        
        logger.info("Cloud Build configuration created")
        
        # Create GitHub Actions workflow
        self._create_github_actions_workflow()
    
    def _create_github_actions_workflow(self):
        """Create GitHub Actions workflow for Cloud Run deployment"""
        workflow_content = f"""
name: Deploy to Cloud Run

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  PROJECT_ID: {self.project_id}
  SERVICE: {self.service_name}
  REGION: {self.region}

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Setup Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{{{ secrets.GCP_SA_KEY }}}}
        project_id: ${{{{ env.PROJECT_ID }}}}
    
    - name: Configure Docker
      run: |
        gcloud auth configure-docker ${{{{ env.REGION }}}}-docker.pkg.dev
    
    - name: Build and Push Container
      run: |
        docker build -t "${{{{ env.REGION }}}}-docker.pkg.dev/${{{{ env.PROJECT_ID }}}}/${{{{ env.SERVICE }}}}/${{{{ env.SERVICE }}}}:${{{{ github.sha }}}}" .
        docker push "${{{{ env.REGION }}}}-docker.pkg.dev/${{{{ env.PROJECT_ID }}}}/${{{{ env.SERVICE }}}}/${{{{ env.SERVICE }}}}:${{{{ github.sha }}}}"
    
    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy ${{{{ env.SERVICE }}}} \\
          --image "${{{{ env.REGION }}}}-docker.pkg.dev/${{{{ env.PROJECT_ID }}}}/${{{{ env.SERVICE }}}}/${{{{ env.SERVICE }}}}:${{{{ github.sha }}}}" \\
          --platform managed \\
          --region ${{{{ env.REGION }}}} \\
          --allow-unauthenticated
    
    - name: Show Service URL
      run: |
        echo "Service deployed to:"
        gcloud run services describe ${{{{ env.SERVICE }}}} \\
          --platform managed \\
          --region ${{{{ env.REGION }}}} \\
          --format 'value(status.url)'
"""
        
        # Create .github/workflows directory if it doesn't exist
        workflow_dir = Path(".github/workflows")
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        # Save workflow file
        workflow_file = workflow_dir / "deploy-cloudrun.yml"
        workflow_file.write_text(workflow_content)
        
        logger.info("GitHub Actions workflow created")
    
    def test_deployment(self, service_url: str) -> bool:
        """
        Test deployed Cloud Run service
        
        Args:
            service_url: Service endpoint URL
            
        Returns:
            Test success status
        """
        logger.info(f"Testing deployment at {service_url}")
        
        try:
            # Test health endpoint
            health_response = requests.get(f"{service_url}/health", timeout=10)
            if health_response.status_code != 200:
                logger.error(f"Health check failed: {health_response.status_code}")
                return False
            
            # Test prediction endpoint
            test_data = {
                "text": "This product is amazing! Great quality and fast delivery.",
                "rating": 5
            }
            
            predict_response = requests.post(
                f"{service_url}/predict",
                json=test_data,
                timeout=10
            )
            
            if predict_response.status_code == 200:
                result = predict_response.json()
                logger.info(f"Prediction successful: {result}")
                return True
            else:
                logger.error(f"Prediction failed: {predict_response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Test request failed: {e}")
            return False
    
    def rollback_deployment(self, revision: str = None):
        """
        Rollback Cloud Run service to previous revision
        
        Args:
            revision: Specific revision to rollback to
        """
        logger.info(f"Rolling back {self.service_name}")
        
        if revision:
            # Rollback to specific revision
            rollback_cmd = [
                "gcloud", "run", "services", "update-traffic",
                self.service_name,
                f"--to-revisions={revision}=100",
                "--platform", "managed",
                "--region", self.region,
                "--project", self.project_id
            ]
        else:
            # Rollback to previous revision
            rollback_cmd = [
                "gcloud", "run", "services", "update-traffic",
                self.service_name,
                "--to-latest",
                "--platform", "managed",
                "--region", self.region,
                "--project", self.project_id
            ]
        
        try:
            subprocess.run(rollback_cmd, check=True)
            logger.info("Rollback completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    def deploy_canary(self, new_image: str, traffic_percentage: int = 10):
        """
        Deploy canary release with gradual traffic shift
        
        Args:
            new_image: New container image
            traffic_percentage: Initial traffic percentage for canary
        """
        logger.info(f"Deploying canary with {traffic_percentage}% traffic")
        
        # Deploy new revision without traffic
        deploy_cmd = [
            "gcloud", "run", "deploy", self.service_name,
            "--image", new_image,
            "--platform", "managed",
            "--region", self.region,
            "--project", self.project_id,
            "--no-traffic",
            "--tag", "canary"
        ]
        
        subprocess.run(deploy_cmd, check=True)
        
        # Split traffic
        traffic_cmd = [
            "gcloud", "run", "services", "update-traffic",
            self.service_name,
            f"--to-tags=canary={traffic_percentage}",
            "--platform", "managed",
            "--region", self.region,
            "--project", self.project_id
        ]
        
        subprocess.run(traffic_cmd, check=True)
        logger.info(f"Canary deployed with {traffic_percentage}% traffic")
        
        return True


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy EchoAI to Cloud Run")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--build-only", action="store_true", help="Only build container")
    parser.add_argument("--deploy-only", help="Deploy existing image")
    parser.add_argument("--allow-unauthenticated", action="store_true", 
                       help="Allow public access")
    parser.add_argument("--setup-cicd", action="store_true", 
                       help="Set up CI/CD pipeline")
    parser.add_argument("--canary", type=int, help="Deploy canary with % traffic")
    parser.add_argument("--rollback", action="store_true", help="Rollback deployment")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = CloudRunDeployer(args.project_id, args.region)
    
    try:
        if args.setup_cicd:
            deployer.setup_continuous_deployment()
            print("CI/CD pipeline configured successfully")
            return
        
        if args.rollback:
            deployer.rollback_deployment()
            print("Rollback completed successfully")
            return
        
        # Build container if not deploy-only
        if args.deploy_only:
            image_url = args.deploy_only
        else:
            image_url = deployer.build_container()
            if args.build_only:
                print(f"Container built: {image_url}")
                return
        
        # Deploy to Cloud Run
        if args.canary:
            deployer.deploy_canary(image_url, args.canary)
            print(f"Canary deployed with {args.canary}% traffic")
        else:
            service_url = deployer.deploy_to_cloudrun(
                image_url, 
                args.allow_unauthenticated
            )
            
            # Set up monitoring
            deployer.setup_monitoring(service_url)
            
            # Test deployment
            if deployer.test_deployment(service_url):
                print(f"✅ Deployment successful!")
                print(f"Service URL: {service_url}")
            else:
                print("⚠️ Deployment completed but tests failed")
                print(f"Service URL: {service_url}")
    
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
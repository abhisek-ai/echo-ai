#!/bin/bash

# deploy-cloudrun.sh
# Automated deployment script for EchoAI to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${PROJECT_ID: "trans-scheme-480511-e3"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"echoai-model-service"}
IMAGE_NAME="echoai"

echo -e "${GREEN}=== EchoAI Cloud Run Deployment ===${NC}"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists gcloud; then
    echo -e "${RED}Error: gcloud CLI not found. Please install Google Cloud SDK${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}Error: Docker not found. Please install Docker${NC}"
    exit 1
fi

# Authenticate with GCP
echo -e "${YELLOW}Authenticating with GCP...${NC}"
gcloud auth list

# Set project
echo -e "${YELLOW}Setting project to $PROJECT_ID...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    containerregistry.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo -e "${YELLOW}Setting up Artifact Registry...${NC}"
gcloud artifacts repositories create cloud-run-source-deploy \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Cloud Run" \
    2>/dev/null || echo "Repository already exists"

# Configure Docker authentication
echo -e "${YELLOW}Configuring Docker authentication...${NC}"
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build container image
echo -e "${GREEN}Building container image...${NC}"
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${IMAGE_NAME}:latest"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found${NC}"
    exit 1
fi

# Build using Cloud Build
echo -e "${YELLOW}Submitting build to Cloud Build...${NC}"
gcloud builds submit --tag $IMAGE_URL --timeout=20m

# Deploy to Cloud Run
echo -e "${GREEN}Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_URL \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 1 \
    --concurrency 100 \
    --port 8080 \
    --allow-unauthenticated \
    --set-env-vars "PROJECT_ID=${PROJECT_ID},MODEL_VERSION=v1.0.0"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --format 'value(status.url)')

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo "Service URL: $SERVICE_URL"

# Test the deployment
echo -e "${YELLOW}Testing deployment...${NC}"

# Test health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" ${SERVICE_URL}/health)

if [ $HEALTH_RESPONSE -eq 200 ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed (HTTP $HEALTH_RESPONSE)${NC}"
fi

# Test prediction endpoint
echo "Testing prediction endpoint..."
PREDICTION_RESPONSE=$(curl -s -X POST ${SERVICE_URL}/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test review", "rating": 5}' \
    -w "\n%{http_code}")

HTTP_CODE=$(echo "$PREDICTION_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$PREDICTION_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq "200" ]; then
    echo -e "${GREEN}✓ Prediction test passed${NC}"
    echo "Response: $RESPONSE_BODY"
else
    echo -e "${RED}✗ Prediction test failed (HTTP $HTTP_CODE)${NC}"
fi

# Set up monitoring
echo -e "${YELLOW}Setting up monitoring...${NC}"

# Create uptime check
gcloud monitoring uptime-checks create $SERVICE_NAME-uptime \
    --display-name="$SERVICE_NAME Uptime Check" \
    --resource-type=uptime-url \
    --hostname=$(echo $SERVICE_URL | sed 's|https://||') \
    --path=/health \
    --check-interval=60s \
    2>/dev/null || echo "Uptime check already exists"

echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Visit your service: $SERVICE_URL"
echo "2. Check metrics: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics?project=$PROJECT_ID"
echo "3. View logs: https://console.cloud.google.com/logs?project=$PROJECT_ID"
echo ""
echo "To deploy updates, run this script again after pushing changes."
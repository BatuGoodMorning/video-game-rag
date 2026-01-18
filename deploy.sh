#!/bin/bash
# Deployment script for Video Game RAG API to Google Cloud Run

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="video-game-rag-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${GREEN}=== Video Game RAG API Deployment ===${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if project ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}Error: Please set GCP_PROJECT_ID environment variable${NC}"
    echo "Example: export GCP_PROJECT_ID=my-project-123"
    exit 1
fi

echo -e "${YELLOW}Project ID:${NC} $PROJECT_ID"
echo -e "${YELLOW}Region:${NC} $REGION"
echo -e "${YELLOW}Service:${NC} $SERVICE_NAME"
echo ""

# Set project
echo -e "${GREEN}[1/6] Setting GCP project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${GREEN}[2/6] Enabling required APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    aiplatform.googleapis.com

# Build Docker image
echo -e "${GREEN}[3/6] Building Docker image...${NC}"
gcloud builds submit --tag $IMAGE_NAME

# Create secrets (if they don't exist)
echo -e "${GREEN}[4/6] Checking secrets...${NC}"
if ! gcloud secrets describe pinecone-api-key &> /dev/null; then
    echo -e "${YELLOW}Creating pinecone-api-key secret...${NC}"
    echo -n "Enter Pinecone API key: "
    read -s PINECONE_KEY
    echo ""
    echo -n "$PINECONE_KEY" | gcloud secrets create pinecone-api-key --data-file=-
fi

if ! gcloud secrets describe google-project-id &> /dev/null; then
    echo -e "${YELLOW}Creating google-project-id secret...${NC}"
    echo -n "$PROJECT_ID" | gcloud secrets create google-project-id --data-file=-
fi

# Deploy Phoenix (tracing UI) if not exists
echo -e "${GREEN}[5/6] Checking Phoenix deployment...${NC}"
if ! gcloud run services describe phoenix --region=$REGION &> /dev/null; then
    echo -e "${YELLOW}Deploying Phoenix tracing UI...${NC}"
    gcloud run deploy phoenix \
        --image arizephoenix/phoenix:latest \
        --region $REGION \
        --platform managed \
        --allow-unauthenticated \
        --memory 1Gi \
        --min-instances 0 \
        --max-instances 1 \
        --port 6006
    
    PHOENIX_URL=$(gcloud run services describe phoenix --region=$REGION --format='value(status.url)')
    echo -e "${GREEN}Phoenix deployed at: $PHOENIX_URL${NC}"
else
    PHOENIX_URL=$(gcloud run services describe phoenix --region=$REGION --format='value(status.url)')
    echo -e "${GREEN}Phoenix already deployed at: $PHOENIX_URL${NC}"
fi

# Deploy main API
echo -e "${GREEN}[6/6] Deploying main API...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --concurrency 20 \
    --timeout 300 \
    --set-env-vars "USE_VERTEX_AI=true,ENABLE_TRACING=true,LOG_LEVEL=INFO,PHOENIX_ENDPOINT=$PHOENIX_URL" \
    --set-secrets "PINECONE_API_KEY=pinecone-api-key:latest,GOOGLE_PROJECT_ID=google-project-id:latest"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo -e "${GREEN}=== Deployment Complete! ===${NC}"
echo ""
echo -e "${GREEN}API URL:${NC} $SERVICE_URL"
echo -e "${GREEN}Phoenix URL:${NC} $PHOENIX_URL"
echo ""
echo -e "${YELLOW}Test the API:${NC}"
echo "curl $SERVICE_URL/health"
echo ""
echo -e "${YELLOW}View logs:${NC}"
echo "gcloud run logs tail $SERVICE_NAME --region=$REGION"
echo ""
echo -e "${YELLOW}View Phoenix traces:${NC}"
echo "Open $PHOENIX_URL in your browser"


# Deployment Guide - Video Game RAG API

Production deployment guide for Google Cloud Platform.

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
3. **Docker** installed (for local testing)
4. **API Keys**:
   - Pinecone API key
   - Google API key (if using Gemini API) OR GCP project with Vertex AI enabled

## Quick Start

### 1. Setup GCP Project

```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"

# Authenticate
gcloud auth login
gcloud config set project $GCP_PROJECT_ID

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    aiplatform.googleapis.com
```

### 2. Create Secrets

```bash
# Pinecone API key
echo -n "your-pinecone-api-key" | gcloud secrets create pinecone-api-key --data-file=-

# Google Project ID (for Vertex AI)
echo -n "$GCP_PROJECT_ID" | gcloud secrets create google-project-id --data-file=-

# Grant Cloud Run access to secrets
PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT_ID --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding pinecone-api-key \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding google-project-id \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 3. Deploy Using Script

```bash
# Run deployment script
./deploy.sh
```

Or manually:

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/video-game-rag-api

# Deploy Phoenix (tracing UI)
gcloud run deploy phoenix \
    --image arizephoenix/phoenix:latest \
    --region $GCP_REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --min-instances 0 \
    --max-instances 1 \
    --port 6006

# Get Phoenix URL
PHOENIX_URL=$(gcloud run services describe phoenix --region=$GCP_REGION --format='value(status.url)')

# Deploy main API
gcloud run deploy video-game-rag-api \
    --image gcr.io/$GCP_PROJECT_ID/video-game-rag-api \
    --region $GCP_REGION \
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
```

### 4. Test Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe video-game-rag-api --region=$GCP_REGION --format='value(status.url)')

# Health check
curl $SERVICE_URL/health

# Test query
curl -X POST $SERVICE_URL/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best RPG games on Nintendo Switch?",
    "top_k": 5,
    "platform": "Switch",
    "use_agent": true
  }'
```

## Configuration

### Environment Variables

Set in Cloud Run deployment:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `USE_VERTEX_AI` | Use Vertex AI instead of Gemini API | `false` | No |
| `ENABLE_TRACING` | Enable OpenTelemetry tracing | `true` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `PHOENIX_ENDPOINT` | Phoenix collector URL | - | No |
| `RATE_LIMIT_CALLS` | Rate limit calls per period | `100` | No |
| `RATE_LIMIT_PERIOD` | Rate limit period (seconds) | `60` | No |

### Secrets (GCP Secret Manager)

| Secret | Description | Required |
|--------|-------------|----------|
| `pinecone-api-key` | Pinecone API key | Yes |
| `google-project-id` | GCP project ID (for Vertex AI) | Yes (if USE_VERTEX_AI=true) |
| `google-api-key` | Google API key (for Gemini API) | Yes (if USE_VERTEX_AI=false) |

## Local Development

### Using Docker Compose

```bash
# Create .env file
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose up

# API available at: http://localhost:8000
# Phoenix UI at: http://localhost:6006
```

### Using Poetry

```bash
# Install dependencies
poetry install

# Run API
poetry run uvicorn src.api.main:app --reload --port 8000

# Run Phoenix (separate terminal)
poetry run phoenix serve
```

## Monitoring

### View Logs

```bash
# Tail logs
gcloud run logs tail video-game-rag-api --region=$GCP_REGION

# Follow logs
gcloud run logs tail video-game-rag-api --region=$GCP_REGION --follow
```

### Phoenix Tracing

Access Phoenix UI at the deployed URL to view:
- LLM calls and latencies
- Token usage
- Prompt/response pairs
- Retrieval performance
- Guardrail triggers

### Cloud Monitoring

```bash
# View metrics in GCP Console
gcloud monitoring dashboards list
```

## CI/CD with Cloud Build

### Setup Trigger

```bash
# Connect GitHub repository
gcloud builds triggers create github \
    --repo-name=video-game-rag \
    --repo-owner=your-username \
    --branch-pattern="^main$" \
    --build-config=cloudbuild.yaml
```

### Manual Build

```bash
gcloud builds submit --config=cloudbuild.yaml
```

## Cost Optimization

### Free Tier Limits

- **Cloud Run**: 2M requests/month, 360K vCPU-seconds, 180K GiB-seconds
- **Cloud Build**: 120 build-minutes/day
- **Secret Manager**: 6 active secret versions
- **Vertex AI**: $300 credit for new accounts

### Tips

1. Set `--min-instances=0` to scale to zero when idle
2. Use `--cpu-throttling=true` to reduce costs
3. Monitor token usage via Phoenix
4. Enable response caching for repeated queries
5. Use Cloud Armor for DDoS protection (if needed)

## Troubleshooting

### Cold Start Issues

```bash
# Set minimum instances
gcloud run services update video-game-rag-api \
    --min-instances=1 \
    --region=$GCP_REGION
```

### Memory Issues

```bash
# Increase memory
gcloud run services update video-game-rag-api \
    --memory=8Gi \
    --region=$GCP_REGION
```

### Secret Access Issues

```bash
# Check IAM permissions
gcloud secrets get-iam-policy pinecone-api-key
```

## Security

### Authentication

To enable authentication:

```bash
# Deploy with authentication required
gcloud run services update video-game-rag-api \
    --no-allow-unauthenticated \
    --region=$GCP_REGION

# Create service account for clients
gcloud iam service-accounts create rag-api-client

# Grant invoke permission
gcloud run services add-iam-policy-binding video-game-rag-api \
    --member="serviceAccount:rag-api-client@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --region=$GCP_REGION
```

### Network Security

- Use VPC Connector for private Pinecone access
- Enable Cloud Armor for rate limiting and DDoS protection
- Use Secret Manager for all sensitive data

## Cleanup

```bash
# Delete services
gcloud run services delete video-game-rag-api --region=$GCP_REGION
gcloud run services delete phoenix --region=$GCP_REGION

# Delete secrets
gcloud secrets delete pinecone-api-key
gcloud secrets delete google-project-id

# Delete images
gcloud container images delete gcr.io/$GCP_PROJECT_ID/video-game-rag-api
```

## Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Open an issue on GitHub
- **GCP Support**: https://cloud.google.com/support


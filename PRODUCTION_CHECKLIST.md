# Production Deployment Checklist

## ✅ Pre-Deployment

### Code & Configuration
- [x] FastAPI backend implemented
- [x] Pydantic Settings for configuration
- [x] LLM factory (Gemini API + Vertex AI support)
- [x] OpenTelemetry + Phoenix tracing
- [x] GCP Secret Manager integration
- [x] Rate limiting middleware
- [x] Caching layer (in-memory LRU)
- [x] Health check endpoints
- [x] Error handling & logging

### Docker & Infrastructure
- [x] Multi-stage Dockerfile
- [x] docker-compose.yml for local dev
- [x] .dockerignore configured
- [x] cloudbuild.yaml for CI/CD
- [x] cloudrun.yaml service config
- [x] deploy.sh automation script

### Documentation
- [x] README_PRODUCTION.md
- [x] DEPLOYMENT.md
- [x] PRODUCTION_CHECKLIST.md
- [x] .env.example template

## 🚀 Deployment Steps

### 1. GCP Setup
```bash
# Set project
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    containerregistry.googleapis.com secretmanager.googleapis.com \
    aiplatform.googleapis.com
```

### 2. Create Secrets
```bash
# Pinecone API key
echo -n "your-pinecone-key" | gcloud secrets create pinecone-api-key --data-file=-

# Google Project ID
echo -n "$GCP_PROJECT_ID" | gcloud secrets create google-project-id --data-file=-

# Grant access
PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT_ID --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding pinecone-api-key \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 3. Deploy
```bash
# Option A: Use deployment script
./deploy.sh

# Option B: Manual deployment
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/video-game-rag-api
gcloud run deploy video-game-rag-api \
    --image gcr.io/$GCP_PROJECT_ID/video-game-rag-api \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --set-env-vars "USE_VERTEX_AI=true,ENABLE_TRACING=true" \
    --set-secrets "PINECONE_API_KEY=pinecone-api-key:latest,GOOGLE_PROJECT_ID=google-project-id:latest"
```

### 4. Verify Deployment
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe video-game-rag-api --region=us-central1 --format='value(status.url)')

# Test health
curl $SERVICE_URL/health

# Test query
curl -X POST $SERVICE_URL/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best RPG games?", "use_agent": true}'
```

## 🔍 Post-Deployment

### Monitoring
- [ ] Check Cloud Run logs: `gcloud run logs tail video-game-rag-api --follow`
- [ ] Access Phoenix UI for LLM traces
- [ ] Verify Cloud Monitoring dashboards
- [ ] Set up alerts for errors and latency

### Performance Testing
- [ ] Load testing with expected traffic
- [ ] Cold start latency measurement
- [ ] Token usage tracking (cost monitoring)
- [ ] Cache hit rate analysis

### Security
- [ ] Review IAM permissions
- [ ] Test rate limiting
- [ ] Verify secret access
- [ ] Enable authentication if needed

## 📊 Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| Request Latency (p95) | < 3s | > 5s |
| Error Rate | < 1% | > 5% |
| Cold Start Time | < 30s | > 60s |
| Memory Usage | < 3GB | > 3.5GB |
| Token Usage/Request | < 2000 | > 5000 |
| Cache Hit Rate | > 30% | < 10% |

## 🐛 Troubleshooting

### Common Issues

**Cold Start Too Slow**
```bash
# Set minimum instances
gcloud run services update video-game-rag-api --min-instances=1
```

**Out of Memory**
```bash
# Increase memory
gcloud run services update video-game-rag-api --memory=8Gi
```

**Secret Access Denied**
```bash
# Check IAM policy
gcloud secrets get-iam-policy pinecone-api-key
```

**High Latency**
- Check Phoenix traces for bottlenecks
- Enable response caching
- Optimize retrieval top_k
- Consider using smaller embedding model

## 💰 Cost Optimization

### Current Configuration
- **Cloud Run**: 4GB RAM, 2 vCPU, 0-10 instances
- **Phoenix**: 1GB RAM, 0-1 instances
- **Vertex AI**: Pay-per-token (Gemini 2.0 Flash)
- **Pinecone**: Free tier (100K vectors)

### Optimization Tips
1. Set `min-instances=0` for low-traffic periods
2. Enable `cpu-throttling=true` to reduce costs
3. Use response caching for repeated queries
4. Monitor token usage via Phoenix
5. Consider batch processing for indexing

### Expected Costs (Monthly)
- **Low Traffic** (<10K requests): $0-5 (free tier)
- **Medium Traffic** (100K requests): $10-30
- **High Traffic** (1M requests): $50-150

## 🔄 CI/CD Pipeline

### GitHub Actions (Optional)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: google-github-actions/setup-gcloud@v1
      - run: gcloud builds submit --config=cloudbuild.yaml
```

### Cloud Build Trigger
```bash
gcloud builds triggers create github \
    --repo-name=video-game-rag \
    --repo-owner=your-username \
    --branch-pattern="^main$" \
    --build-config=cloudbuild.yaml
```

## 📝 Maintenance

### Regular Tasks
- [ ] Weekly: Review error logs
- [ ] Weekly: Check token usage and costs
- [ ] Monthly: Update dependencies
- [ ] Monthly: Review and optimize cache
- [ ] Quarterly: Load testing
- [ ] Quarterly: Security audit

### Backup & Recovery
- Pinecone data is managed (no backup needed)
- Code is in Git (version controlled)
- Secrets in Secret Manager (versioned)
- Cloud Run auto-scales and recovers

## 🎯 Success Criteria

- [x] API deployed and accessible
- [x] Health checks passing
- [x] Tracing operational
- [x] Secrets properly configured
- [x] Rate limiting working
- [x] Error handling tested
- [ ] Load testing completed
- [ ] Monitoring alerts configured
- [ ] Documentation reviewed
- [ ] Team trained on operations

## 📞 Support Contacts

- **GCP Support**: https://cloud.google.com/support
- **Pinecone Support**: https://www.pinecone.io/support
- **Phoenix Docs**: https://docs.arize.com/phoenix
- **Project Issues**: GitHub Issues

---

**Last Updated**: 2026-01-18
**Version**: 1.0.0


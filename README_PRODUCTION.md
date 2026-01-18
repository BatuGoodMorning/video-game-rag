# Video Game RAG - Production API

A production-ready RAG (Retrieval-Augmented Generation) API for video game information with multi-hop reasoning, reranking, and comprehensive observability.

## 🚀 Features

### Core RAG Pipeline
- **Multi-hop Reasoning**: LangGraph agent for complex queries
- **Hybrid Chunking**: Summary, detail, and similarity chunks
- **Reranking**: Cross-encoder for improved precision
- **Guardrails**: Input/output validation and hallucination detection
- **Vector Search**: Pinecone with HNSW algorithm

### Production Features
- **FastAPI Backend**: REST API with async support
- **Vertex AI Integration**: Managed Gemini LLM
- **OpenTelemetry Tracing**: Phoenix + Cloud Trace
- **GCP Secret Manager**: Secure credential management
- **Docker**: Multi-stage builds for optimal image size
- **Cloud Run**: Serverless deployment with auto-scaling
- **Caching**: In-memory LRU cache for embeddings/responses
- **Rate Limiting**: Per-IP request throttling
- **Health Checks**: Liveness and readiness probes

## 📋 Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         FastAPI (Cloud Run)             │
│  ┌─────────────────────────────────┐   │
│  │  OpenTelemetry Instrumentation  │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │    LangGraph Agent              │   │
│  │  ┌──────────┐  ┌──────────┐    │   │
│  │  │Guardrails│  │Multi-hop │    │   │
│  │  └──────────┘  └──────────┘    │   │
│  └─────────────────────────────────┘   │
└──────┬──────────────────────┬──────────┘
       │                      │
       ▼                      ▼
┌─────────────┐      ┌─────────────┐
│  Pinecone   │      │ Vertex AI   │
│ (Vectors)   │      │  (Gemini)   │
└─────────────┘      └─────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Phoenix (Cloud Run) + Cloud Trace  │
│         Observability               │
└─────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | REST API with OpenAPI docs |
| **LLM** | Vertex AI Gemini | Text generation |
| **Vector DB** | Pinecone | Semantic search |
| **Embeddings** | Sentence-Transformers | Local embeddings (free) |
| **Orchestration** | LangChain + LangGraph | RAG pipeline & agent |
| **Tracing** | Phoenix + OpenTelemetry | Observability |
| **Deployment** | Cloud Run | Serverless containers |
| **Secrets** | GCP Secret Manager | Credential management |
| **CI/CD** | Cloud Build | Automated deployment |

## 🚦 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- GCP account (for deployment)
- Pinecone account
- Google API key or Vertex AI access

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/your-username/video-game-rag.git
cd video-game-rag

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run API
poetry run uvicorn src.api.main:app --reload --port 8000

# 5. Access API
# Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

### Docker

```bash
# Build image
docker build -t video-game-rag-api .

# Run container
docker run -p 8000:8000 \
  -e PINECONE_API_KEY=your-key \
  -e GOOGLE_API_KEY=your-key \
  video-game-rag-api

# Or use docker-compose
docker-compose up
```

### Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

```bash
# Quick deploy
export GCP_PROJECT_ID="your-project-id"
./deploy.sh
```

## 📡 API Endpoints

### Query

```bash
POST /api/v1/query
```

**Request:**
```json
{
  "query": "What are the best RPG games on Nintendo Switch?",
  "top_k": 5,
  "platform": "Switch",
  "genre": "RPG",
  "use_reranker": true,
  "use_agent": true
}
```

**Response:**
```json
{
  "answer": "Based on the game database, here are some excellent RPG games...",
  "sources": ["The Legend of Zelda: Breath of the Wild", "Xenoblade Chronicles 3"],
  "query_type": "simple",
  "retrieval_latency_ms": 145.2,
  "total_latency_ms": 1823.5,
  "reranked": true,
  "guardrail_status": "passed",
  "chunks_retrieved": 5
}
```

### Health Checks

```bash
GET /health          # Liveness probe
GET /ready           # Readiness probe
GET /health/components  # Detailed component status
```

### Admin

```bash
GET /api/v1/admin/stats        # System statistics
GET /api/v1/admin/cache/stats  # Cache statistics
POST /api/v1/admin/cache/clear # Clear cache
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
# Pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=video-games

# Vertex AI (recommended for production)
USE_VERTEX_AI=true
GOOGLE_PROJECT_ID=your-gcp-project
GOOGLE_LOCATION=us-central1

# OR Gemini API (for development)
USE_VERTEX_AI=false
GOOGLE_API_KEY=your-google-api-key

# Tracing
ENABLE_TRACING=true
PHOENIX_ENDPOINT=http://localhost:6006

# API
LOG_LEVEL=INFO
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60

# GCP Secret Manager (optional)
USE_SECRET_MANAGER=false
```

## 📊 Monitoring & Observability

### Phoenix Tracing

Access Phoenix UI to view:
- **LLM Calls**: Latency, tokens, cost
- **Prompts & Responses**: Full conversation history
- **Retrieval Performance**: Similarity scores, reranking
- **Guardrails**: Input/output validation results

### Cloud Monitoring

- Request latency histograms
- Error rates and status codes
- Resource utilization (CPU, memory)
- Custom metrics via OpenTelemetry

### Logs

```bash
# Local
tail -f logs/api.log

# Cloud Run
gcloud run logs tail video-game-rag-api --follow
```

## 🧪 Testing

```bash
# Run tests
poetry run pytest

# Test API locally
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about Elden Ring", "use_agent": true}'

# Load testing
poetry run locust -f tests/load_test.py
```

## 📦 Project Structure

```
video_game_rag/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # App entry point
│   │   ├── routes/            # API endpoints
│   │   ├── schemas.py         # Pydantic models
│   │   ├── dependencies.py    # DI & singletons
│   │   ├── middleware.py      # CORS, logging, rate limiting
│   │   ├── tracing.py         # OpenTelemetry setup
│   │   ├── secrets.py         # GCP Secret Manager
│   │   └── cache.py           # Caching layer
│   ├── llm/                   # LLM factory (Gemini/Vertex AI)
│   ├── rag/                   # RAG pipeline
│   │   ├── agent.py           # LangGraph agent
│   │   ├── chain.py           # Simple RAG chain
│   │   ├── retriever.py       # Vector search
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── embeddings/            # Embedding generation
│   ├── vectorstores/          # Pinecone integration
│   ├── chunking/              # Hybrid chunking strategies
│   └── config.py              # Pydantic Settings
├── scripts/                   # Data fetching & indexing
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Local development
├── cloudbuild.yaml            # CI/CD pipeline
├── cloudrun.yaml              # Cloud Run config
├── deploy.sh                  # Deployment script
└── pyproject.toml             # Poetry dependencies
```

## 🔐 Security

- **Secret Management**: GCP Secret Manager for API keys
- **Rate Limiting**: Per-IP throttling (100 req/min default)
- **Input Validation**: Pydantic schemas
- **Guardrails**: Off-topic query filtering
- **CORS**: Configurable allowed origins
- **Non-root User**: Docker container runs as appuser

## 💰 Cost Estimation

### Free Tier (Monthly)

- **Cloud Run**: 2M requests, 360K vCPU-sec, 180K GiB-sec
- **Vertex AI**: $300 credit (new accounts)
- **Pinecone**: 1 index, 100K vectors
- **Cloud Build**: 120 build-minutes/day
- **Secret Manager**: 6 active versions

**Estimated cost for hobby project**: $0-5/month

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [API Documentation](http://your-api-url/docs)
- [Architecture Deep Dive](docs/architecture.md)

## 🙏 Acknowledgments

- LangChain & LangGraph for orchestration
- Arize Phoenix for LLM observability
- Pinecone for vector search
- Google for Vertex AI/Gemini


# Video Game RAG

A Retrieval-Augmented Generation (RAG) system for video game information, featuring dual vector database comparison (Pinecone with HNSW vs Qdrant with IVF+PQ).

## Features

- **Data Collection**: Fetches game info from Wikipedia API (PC, PS5, Nintendo Switch)
- **Chunking**: Hybrid strategy with summary chunks + overlapping detail chunks
- **Embeddings**: HuggingFace sentence-transformers (local, free)
- **Vector Stores**: 
  - Pinecone (HNSW algorithm)
  - Qdrant (IVF + Product Quantization)
- **RAG Pipeline**: LangChain + Gemini LLM
- **LangGraph Agent**: Multi-hop reasoning + guardrails
- **Streamlit UI**: Chat interface with vector DB comparison

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Wikipedia API  │────▶│   Chunking      │────▶│   Embeddings    │
└─────────────────┘     │  (Summary +     │     │ (sentence-      │
                        │   Detail)       │     │  transformers)  │
                        └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                │                                │
                        ▼                                ▼                                │
                ┌───────────────┐                ┌───────────────┐                        │
                │   Pinecone    │                │    Qdrant     │                        │
                │    (HNSW)     │                │  (IVF + PQ)   │                        │
                └───────┬───────┘                └───────┬───────┘                        │
                        │                                │                                │
                        └────────────────┬───────────────┘                                │
                                         │                                                │
                                         ▼                                                │
                                ┌─────────────────┐                                       │
                                │ LangGraph Agent │◀──────────────────────────────────────┘
                                │  + Guardrails   │
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │   Gemini LLM    │
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  Streamlit UI   │
                                └─────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
# Install Poetry if not installed
pip install poetry

# Install project dependencies
poetry install
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Pinecone (Vector Database - HNSW)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=video-games

# Qdrant (Vector Database - IVF+PQ)
# For cloud:
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-cluster.qdrant.io
# For local: leave empty and run Qdrant locally

# Google Gemini (LLM)
GOOGLE_API_KEY=your_google_api_key
```

### 3. Fetch Game Data

```bash
poetry run python scripts/fetch_games.py
```

This fetches ~300 games from Wikipedia (100 per platform).

### 4. Index Data

```bash
poetry run python scripts/index_data.py
```

This creates embeddings and indexes them to both vector stores.

### 5. Run the App

```bash
poetry run streamlit run src/app.py
```

## Project Structure

```
video_game_rag/
├── pyproject.toml          # Poetry dependencies
├── .env                    # API keys (not in git)
├── .env.example            # API key template
├── .gitignore
├── README.md
├── data/                   # Game data (raw + processed)
├── src/
│   ├── config.py           # Environment config
│   ├── data/
│   │   ├── wikipedia_client.py  # Wikipedia API client
│   │   └── processor.py         # Data cleaning
│   ├── chunking/
│   │   └── strategies.py        # Chunking logic
│   ├── embeddings/
│   │   └── embed.py             # Embedding generation
│   ├── vectorstores/
│   │   ├── pinecone_store.py    # Pinecone HNSW
│   │   └── qdrant_store.py      # Qdrant IVF+PQ
│   ├── rag/
│   │   ├── retriever.py         # Unified retriever
│   │   ├── chain.py             # RAG chain
│   │   └── agent.py             # LangGraph agent
│   └── app.py                   # Streamlit UI
└── scripts/
    ├── fetch_games.py           # Data fetching
    └── index_data.py            # Indexing
```

## Vector Store Comparison

| Feature | Pinecone (HNSW) | Qdrant (IVF+PQ) |
|---------|-----------------|-----------------|
| Algorithm | Hierarchical Navigable Small World | Inverted File Index + Product Quantization |
| Memory | Higher (full vectors) | Lower (compressed) |
| Speed | Generally faster | Slightly slower due to decompression |
| Recall | Higher | May have slight recall loss |
| Cost | Serverless pricing | Self-hosted or cloud |

## LangGraph Agent

The agent implements:

1. **Input Guardrail**: Filters off-topic queries
2. **Query Router**: Routes simple vs complex queries
3. **Multi-hop Reasoning**: For comparison/recommendation queries
4. **Output Guardrail**: Validates responses for hallucinations

## Technologies Used

- **LangChain**: RAG pipeline, prompt templates
- **LangGraph**: Agent workflow
- **LlamaIndex**: Document processing (optional)
- **Sentence-Transformers**: Local embeddings
- **Pinecone**: Vector database (HNSW)
- **Qdrant**: Vector database (IVF+PQ)
- **Gemini**: LLM for generation
- **Streamlit**: UI

## License

MIT


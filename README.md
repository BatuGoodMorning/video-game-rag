# Video Game RAG

A Retrieval-Augmented Generation (RAG) system for video game information with multi-hop reasoning, reranking, and comprehensive evaluation.

## Features

- **Data Collection**: Fetches game info from Wikipedia API (PC, PS5, Nintendo Switch)
- **Chunking**: Hybrid strategy with summary, detail, and similarity chunks for multi-hop reasoning
- **Embeddings**: HuggingFace sentence-transformers (local, free)
- **Vector Store**: Pinecone (HNSW algorithm)
- **Reranker**: Cross-encoder reranking for improved precision
- **RAG Pipeline**: LangChain + Gemini LLM
- **LangGraph Agent**: Multi-hop reasoning with similarity chunks + guardrails
- **Evaluation**: NDCG, MRR, Hit Rate, Precision@K, Recall@K metrics
- **TruLens**: Groundedness, relevance, and hallucination detection
- **Streamlit UI**: Chat interface with reranker toggle

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Wikipedia API  │────▶│   Chunking      │────▶│   Embeddings    │
└─────────────────┘     │  (Summary +     │     │ (sentence-      │
                        │   Detail +      │     │  transformers)  │
                        │   Similarity)   │     └────────┬────────┘
                        └─────────────────┘              │
                                                         ▼
                                                ┌───────────────┐
                                                │   Pinecone    │
                                                │    (HNSW)     │
                                                └───────┬───────┘
                                                        │
                                                        ▼
                                                ┌───────────────┐
                                                │   Reranker    │
                                                │ (CrossEncoder)│
                                                └───────┬───────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │ LangGraph Agent │
                                                │  + Multi-hop    │
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
# Pinecone (Vector Database)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=video-games

# Google Gemini (LLM)
GOOGLE_API_KEY=your_google_api_key
```

### 3. Fetch Game Data

```bash
poetry run python scripts/fetch_games.py
```

This fetches best-selling games from Wikipedia (PC, PS4/PS5, Nintendo Switch).

### 4. Index Data

```bash
poetry run python scripts/index_data.py
```

This creates embeddings and indexes them to Pinecone, including:
- Summary chunks (~200 tokens each)
- Detail chunks (~512 tokens with overlap)
- Similarity chunks (for multi-hop reasoning)

### 5. Run the App

```bash
poetry run streamlit run src/app.py
```

### 6. Run Evaluation (Optional)

```bash
poetry run python scripts/evaluate.py
```

This generates a synthetic dataset and evaluates retrieval quality with and without reranking.

## Project Structure

```
video_game_rag/
├── pyproject.toml          # Poetry dependencies
├── .env                    # API keys (not in git)
├── README.md
├── data/                   # Game data (raw + processed)
│   ├── raw/
│   ├── processed/
│   └── evaluation/         # Evaluation reports
├── src/
│   ├── config.py           # Environment config
│   ├── data/
│   │   ├── wikipedia_client.py  # Wikipedia API client
│   │   └── processor.py         # Data cleaning
│   ├── chunking/
│   │   └── strategies.py        # Hybrid chunking (summary + detail + similarity)
│   ├── embeddings/
│   │   └── embed.py             # Embedding generation
│   ├── vectorstores/
│   │   └── pinecone_store.py    # Pinecone HNSW
│   ├── rag/
│   │   ├── retriever.py         # Retriever with reranker
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── chain.py             # RAG chain
│   │   └── agent.py             # LangGraph multi-hop agent
│   ├── evaluation/              # Evaluation framework
│   │   ├── metrics.py           # NDCG, MRR, Precision, Recall
│   │   ├── dataset.py           # Synthetic dataset generation
│   │   ├── evaluator.py         # Evaluation pipeline
│   │   └── trulens_eval.py      # Groundedness, hallucination detection
│   └── app.py                   # Streamlit UI
└── scripts/
    ├── fetch_games.py           # Data fetching
    ├── index_data.py            # Indexing with similarity chunks
    └── evaluate.py              # Run evaluation
```

## Key Components

### Chunking Strategy

The system uses three types of chunks:

1. **Summary Chunks** (~200 tokens): Quick facts for simple queries
2. **Detail Chunks** (~512 tokens, 100 overlap): Full descriptions, gameplay, plot
3. **Similarity Chunks**: Describe relationships between games for multi-hop reasoning

### Reranker

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters) to rerank initial retrieval results:

- Initial retrieval: Top 30 from Pinecone
- Reranking: Cross-encoder scores each (query, chunk) pair
- Final output: Top 5 highest-scoring chunks

### Multi-hop Reasoning

For complex queries like "games similar to The Witcher 3":

1. **Hop 1**: Retrieve info about reference game (The Witcher 3)
2. **Hop 2**: Fetch similarity chunks listing related games
3. **Hop 3**: Get details about each similar game
4. **Synthesis**: Combine all context for final answer

### Evaluation Metrics

**Retrieval Metrics:**
- Precision@K: Relevant items in top-K / K
- Recall@K: Relevant items found / total relevant
- Hit Rate@K: Any relevant item in top-K?
- MRR: Mean Reciprocal Rank of first relevant
- NDCG@K: Normalized Discounted Cumulative Gain

**TruLens Metrics:**
- Groundedness: Is answer supported by context?
- Context Relevance: Is retrieved context relevant to query?
- Answer Relevance: Does answer address the query?
- No Hallucination: Are claims supported by context?

## LangGraph Agent

The agent implements a state machine with:

1. **Input Guardrail**: Filters off-topic queries
2. **Query Router**: Routes simple vs complex queries
3. **Simple RAG Node**: Direct retrieval + answer
4. **Multi-hop Node**: Multiple retrieval steps with similarity chunks
5. **Output Guardrail**: Validates responses for hallucinations

## Technologies Used

- **LangChain**: RAG pipeline, prompt templates
- **LangGraph**: Agent workflow
- **Sentence-Transformers**: Local embeddings + cross-encoder reranking
- **Pinecone**: Vector database (HNSW)
- **Gemini**: LLM for generation
- **Streamlit**: UI

## License

MIT

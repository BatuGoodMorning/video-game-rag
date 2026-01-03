#!/usr/bin/env python3
"""Script to chunk, embed, and index game data into vector stores."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.processor import GameDataProcessor
from src.chunking.strategies import GameChunker
from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore
from src.vectorstores.qdrant_store import QdrantStore


def main():
    """Index game data into vector stores."""
    print("=" * 60)
    print("Video Game RAG - Data Indexer")
    print("=" * 60)
    
    # Validate config
    missing = config.validate()
    if "PINECONE_API_KEY" in missing:
        print("Warning: Pinecone API key not set, skipping Pinecone indexing")
    if "GOOGLE_API_KEY" in missing:
        print("Warning: Google API key not set, Gemini fallback won't work")
    
    # Load processed data
    print("\n1. Loading processed game data...")
    processor = GameDataProcessor(config.DATA_DIR)
    
    try:
        games = processor.load_processed_data()
        print(f"   Loaded {len(games)} games")
    except FileNotFoundError:
        print("   Error: No processed data found!")
        print("   Run 'python scripts/fetch_games.py' first")
        return
    
    # Create chunks
    print("\n2. Creating chunks...")
    chunker = GameChunker(
        summary_max_tokens=config.SUMMARY_CHUNK_SIZE,
        detail_chunk_size=config.DETAIL_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    
    chunks = chunker.chunk_all_games(games)
    stats = chunker.get_stats(chunks)
    print(f"   Created {stats['total_chunks']} chunks")
    print(f"   - Summary chunks: {stats['summary_chunks']}")
    print(f"   - Detail chunks: {stats['detail_chunks']}")
    print(f"   - Unique games: {stats['unique_games']}")
    
    # Generate embeddings
    print("\n3. Generating embeddings...")
    embedder = EmbeddingGenerator(
        model_key="mpnet",
        use_gpu=True,
        google_api_key=config.GOOGLE_API_KEY,
    )
    
    print(f"   Using model: {embedder.model_info['name']}")
    print(f"   Device: {embedder.device}")
    
    chunks_list, embeddings = embedder.embed_chunks(chunks, show_progress=True)
    print(f"   Generated {len(embeddings)} embeddings of dimension {embedder.dimension}")
    
    # Index to Pinecone
    if config.PINECONE_API_KEY:
        print("\n4. Indexing to Pinecone (HNSW)...")
        try:
            pinecone_store = PineconeStore(
                api_key=config.PINECONE_API_KEY,
                index_name=config.PINECONE_INDEX_NAME,
                dimension=embedder.dimension,
            )
            
            count = pinecone_store.upsert_chunks(chunks_list, embeddings)
            print(f"   Upserted {count} vectors to Pinecone")
            
            stats = pinecone_store.get_stats()
            print(f"   Total vectors in index: {stats['total_vectors']}")
        except Exception as e:
            print(f"   Error indexing to Pinecone: {e}")
    else:
        print("\n4. Skipping Pinecone (no API key)")
    
    # Index to Qdrant
    print("\n5. Indexing to Qdrant (IVF + PQ)...")
    try:
        # Try cloud first, fall back to local
        qdrant_store = QdrantStore(
            url=config.QDRANT_URL if config.QDRANT_API_KEY else None,
            api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None,
            collection_name="video_games",
            dimension=embedder.dimension,
            use_local=not config.QDRANT_API_KEY and not config.QDRANT_URL,
        )
        
        qdrant_store.create_collection(recreate=True)
        count = qdrant_store.upsert_chunks(chunks_list, embeddings)
        print(f"   Upserted {count} vectors to Qdrant")
        
        stats = qdrant_store.get_stats()
        print(f"   Total vectors: {stats['total_vectors']}")
        print(f"   Status: {stats['status']}")
    except Exception as e:
        print(f"   Error indexing to Qdrant: {e}")
        print("   Make sure Qdrant is running locally or provide cloud credentials")
    
    print("\n" + "=" * 60)
    print("Indexing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


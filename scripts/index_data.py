#!/usr/bin/env python3
"""Script to chunk, embed, and index game data into Pinecone."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.processor import GameDataProcessor
from src.chunking.strategies import GameChunker
from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore


def main():
    """Index game data into Pinecone."""
    print("=" * 60)
    print("Video Game RAG - Data Indexer")
    print("=" * 60)
    
    # Validate config
    missing = config.validate()
    if "PINECONE_API_KEY" in missing:
        print("Error: Pinecone API key not set!")
        print("Please set PINECONE_API_KEY in your .env file")
        return
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
    
    # Create all chunk types including similarity chunks
    chunks = chunker.chunk_all_games(games)
    
    # Also create similarity chunks for multi-hop reasoning
    print("   Creating similarity chunks for multi-hop reasoning...")
    similarity_chunks = chunker.create_similarity_chunks(games)
    chunks.extend(similarity_chunks)
    
    stats = chunker.get_stats(chunks)
    print(f"   Created {stats['total_chunks']} chunks")
    print(f"   - Summary chunks: {stats['summary_chunks']}")
    print(f"   - Detail chunks: {stats['detail_chunks']}")
    print(f"   - Similarity chunks: {stats['similarity_chunks']}")
    print(f"   - Unique games: {stats['unique_games']}")
    
    # Generate embeddings (with cache)
    print("\n3. Generating embeddings...")
    embedder = EmbeddingGenerator(
        model_key="mpnet",
        use_gpu=True,
        google_api_key=config.GOOGLE_API_KEY,
    )
    
    print(f"   Using model: {embedder.model_info['name']}")
    print(f"   Device: {embedder.device}")
    
    # Check for embedding cache
    import pickle
    embedding_cache_path = config.DATA_DIR / "processed" / "embeddings_cache.pkl"
    
    if embedding_cache_path.exists():
        print("   Checking embedding cache...")
        try:
            with open(embedding_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # Verify cache is valid (same number of chunks and model)
                if (cached_data.get('chunk_count') == len(chunks) and 
                    cached_data.get('model') == embedder.model_info['name']):
                    chunks_list = cached_data['chunks']
                    embeddings = cached_data['embeddings']
                    print(f"   ✓ Loaded {len(embeddings)} embeddings from cache")
                else:
                    print("   Cache outdated, regenerating...")
                    chunks_list, embeddings = embedder.embed_chunks(chunks, show_progress=True)
                    # Save cache
                    with open(embedding_cache_path, 'wb') as f:
                        pickle.dump({
                            'chunks': chunks_list,
                            'embeddings': embeddings,
                            'chunk_count': len(chunks),
                            'model': embedder.model_info['name']
                        }, f)
                    print(f"   Generated {len(embeddings)} embeddings of dimension {embedder.dimension}")
        except Exception as e:
            print(f"   Cache read error: {e}, regenerating...")
            chunks_list, embeddings = embedder.embed_chunks(chunks, show_progress=True)
            # Save cache
            with open(embedding_cache_path, 'wb') as f:
                pickle.dump({
                    'chunks': chunks_list,
                    'embeddings': embeddings,
                    'chunk_count': len(chunks),
                    'model': embedder.model_info['name']
                }, f)
            print(f"   Generated {len(embeddings)} embeddings of dimension {embedder.dimension}")
    else:
        chunks_list, embeddings = embedder.embed_chunks(chunks, show_progress=True)
        # Save cache
        with open(embedding_cache_path, 'wb') as f:
            pickle.dump({
                'chunks': chunks_list,
                'embeddings': embeddings,
                'chunk_count': len(chunks),
                'model': embedder.model_info['name']
            }, f)
        print(f"   Generated {len(embeddings)} embeddings of dimension {embedder.dimension}")
        print(f"   Cache saved for next time")
    
    # Index to Pinecone
    print("\n4. Indexing to Pinecone (HNSW)...")
    try:
        pinecone_store = PineconeStore(
            api_key=config.PINECONE_API_KEY,
            index_name=config.PINECONE_INDEX_NAME,
            dimension=embedder.dimension,
        )
        
        # Delete all existing vectors to avoid conflicts
        print("   Clearing existing vectors...")
        try:
            pinecone_store.delete_all()
        except Exception as e:
            print(f"   Note: {e}")
        
        count = pinecone_store.upsert_chunks(chunks_list, embeddings)
        print(f"   Upserted {count} vectors to Pinecone")
        
        stats = pinecone_store.get_stats()
        print(f"   Total vectors in index: {stats['total_vectors']}")
    except Exception as e:
        print(f"   Error indexing to Pinecone: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Indexing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

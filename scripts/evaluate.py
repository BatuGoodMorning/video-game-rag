#!/usr/bin/env python3
"""Script to run RAG evaluation with metrics and TruLens feedback."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.processor import GameDataProcessor
from src.embeddings.embed import EmbeddingGenerator
from src.vectorstores.pinecone_store import PineconeStore
from src.rag.retriever import GameRetriever
from src.rag.reranker import Reranker
from src.rag.chain import RAGChain
from src.evaluation.dataset import SyntheticDatasetGenerator
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.trulens_eval import TruLensEvaluator


def main():
    """Run full RAG evaluation pipeline."""
    print("=" * 60)
    print("Video Game RAG - Evaluation Pipeline")
    print("=" * 60)
    
    # Validate config
    missing = config.validate()
    if missing:
        print(f"Error: Missing API keys: {', '.join(missing)}")
        return
    
    # Setup paths
    eval_dir = config.DATA_DIR / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load game data
    print("\n1. Loading game data...")
    processor = GameDataProcessor(config.DATA_DIR)
    
    try:
        games = processor.load_processed_data()
        print(f"   Loaded {len(games)} games")
    except FileNotFoundError:
        print("   Error: No processed data found!")
        print("   Run 'python scripts/fetch_games.py' first")
        return
    
    # 2. Generate or load evaluation dataset
    print("\n2. Preparing evaluation dataset...")
    dataset_path = eval_dir / "eval_dataset.json"
    
    if dataset_path.exists():
        print(f"   Loading existing dataset from {dataset_path}")
        generator = SyntheticDatasetGenerator(games, use_llm=False)
        queries = generator.load_dataset(dataset_path)
        print(f"   Loaded {len(queries)} queries")
    else:
        print("   Generating synthetic evaluation dataset...")
        generator = SyntheticDatasetGenerator(
            games,
            google_api_key=config.GOOGLE_API_KEY,
            use_llm=True,
        )
        
        queries = generator.generate_full_dataset(
            simple=30,
            comparison=20,
            complex=15,
            llm_generated=15,
        )
        
        # Save dataset
        generator.save_dataset(queries, dataset_path)
        print(f"   Generated and saved {len(queries)} queries to {dataset_path}")
    
    # 3. Initialize RAG components
    print("\n3. Initializing RAG components...")
    
    embedder = EmbeddingGenerator(
        model_key="mpnet",
        use_gpu=True,
        google_api_key=config.GOOGLE_API_KEY,
    )
    print(f"   Embedder: {embedder.model_info['name']} on {embedder.device}")
    
    pinecone_store = PineconeStore(
        api_key=config.PINECONE_API_KEY,
        index_name=config.PINECONE_INDEX_NAME,
        dimension=embedder.dimension,
    )
    stats = pinecone_store.get_stats()
    print(f"   Pinecone: {stats['total_vectors']} vectors")
    
    reranker = Reranker()
    print(f"   Reranker: {reranker.model_name}")
    
    retriever = GameRetriever(
        embedding_generator=embedder,
        pinecone_store=pinecone_store,
        reranker=reranker,
    )
    
    # 4. Run retrieval evaluation
    print("\n4. Running retrieval evaluation...")
    evaluator = RAGEvaluator(retriever, k_values=[3, 5, 10])
    
    # Compare with and without reranker
    print("\n   Comparing retrieval with and without reranker...")
    report_without, report_with, improvement = evaluator.compare_with_without_reranker(
        queries, k=5, show_progress=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nWithout Reranker:")
    for name, value in sorted(report_without.aggregated_metrics.items()):
        print(f"  {name}: {value:.4f}")
    print(f"  avg_latency: {report_without.avg_latency_ms:.2f}ms")
    
    print("\nWith Reranker:")
    for name, value in sorted(report_with.aggregated_metrics.items()):
        print(f"  {name}: {value:.4f}")
    print(f"  avg_latency: {report_with.avg_latency_ms:.2f}ms")
    
    print("\nImprovement from Reranker:")
    for name, value in sorted(improvement.items()):
        print(f"  {name}: {value:+.4f}")
    
    # Save retrieval report
    retrieval_report_path = eval_dir / f"retrieval_report_{timestamp}.json"
    evaluator.save_report(report_with, retrieval_report_path)
    print(f"\nRetrieval report saved to {retrieval_report_path}")
    
    # 5. Run TruLens evaluation on a sample
    print("\n5. Running TruLens evaluation (sample)...")
    trulens_evaluator = TruLensEvaluator(google_api_key=config.GOOGLE_API_KEY)
    
    # Initialize RAG chain for answer generation
    rag_chain = RAGChain(retriever, google_api_key=config.GOOGLE_API_KEY)
    
    # Evaluate a sample of queries
    sample_size = min(10, len(queries))
    sample_queries = queries[:sample_size]
    
    trulens_results = []
    for i, query in enumerate(sample_queries):
        print(f"   Evaluating query {i+1}/{sample_size}: {query.query[:50]}...")
        
        try:
            # Get RAG response
            result = rag_chain.query(query.query, top_k=5, use_reranker=True)
            
            # Extract context texts
            context = [
                chunk.get("metadata", {}).get("text", "")
                for chunk in result.get("chunks", [])
            ]
            
            # Run TruLens evaluation
            trulens_result = trulens_evaluator.evaluate_response(
                query=query.query,
                answer=result["answer"],
                context=context,
            )
            trulens_results.append(trulens_result)
            
            # Print individual scores
            scores_str = ", ".join([
                f"{fb.name}: {fb.score:.2f}"
                for fb in trulens_result.feedbacks
            ])
            print(f"      Scores: {scores_str}")
            
        except Exception as e:
            print(f"      Error: {e}")
            continue
    
    # Aggregate TruLens scores
    if trulens_results:
        print("\n" + "=" * 60)
        print("TRULENS EVALUATION RESULTS")
        print("=" * 60)
        
        aggregate_scores = trulens_evaluator.get_aggregate_scores(trulens_results)
        for name, score in sorted(aggregate_scores.items()):
            print(f"  avg_{name}: {score:.4f}")
        
        # Save TruLens results
        trulens_report_path = eval_dir / f"trulens_report_{timestamp}.json"
        trulens_data = {
            "aggregate_scores": aggregate_scores,
            "individual_results": [r.to_dict() for r in trulens_results],
        }
        with open(trulens_report_path, "w", encoding="utf-8") as f:
            json.dump(trulens_data, f, indent=2, ensure_ascii=False)
        print(f"\nTruLens report saved to {trulens_report_path}")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries evaluated: {len(queries)}")
    print(f"Retrieval metrics computed: ✓")
    print(f"Reranker comparison: ✓")
    print(f"TruLens sample size: {len(trulens_results)}")
    print(f"\nReports saved to: {eval_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""RAG Evaluation pipeline.

Runs evaluation on the RAG system using the synthetic dataset
and computes all retrieval and generation metrics.
"""

import time
import json
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm

from src.config import config
from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.dataset import EvalQuery
from src.rag.retriever import GameRetriever


@dataclass
class EvaluationResult:
    """Results from a single query evaluation."""
    
    query: str
    query_type: str
    retrieved_games: list[str]
    relevant_games: list[str]
    metrics: dict[str, float]
    latency_ms: float
    reranked: bool
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "retrieved_games": self.retrieved_games,
            "relevant_games": self.relevant_games,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "reranked": self.reranked,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    
    total_queries: int
    aggregated_metrics: dict[str, float]
    metrics_by_type: dict[str, dict[str, float]]
    avg_latency_ms: float
    reranker_improvement: Optional[dict[str, float]] = None
    individual_results: list[EvaluationResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "aggregated_metrics": self.aggregated_metrics,
            "metrics_by_type": self.metrics_by_type,
            "avg_latency_ms": self.avg_latency_ms,
            "reranker_improvement": self.reranker_improvement,
            "individual_results": [r.to_dict() for r in self.individual_results],
        }
    
    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            "=" * 60,
            "RAG Evaluation Report",
            "=" * 60,
            f"Total Queries: {self.total_queries}",
            f"Average Latency: {self.avg_latency_ms:.2f}ms",
            "",
            "Aggregated Metrics:",
        ]
        
        for name, value in sorted(self.aggregated_metrics.items()):
            lines.append(f"  {name}: {value:.4f}")
        
        if self.metrics_by_type:
            lines.append("")
            lines.append("Metrics by Query Type:")
            for qtype, metrics in self.metrics_by_type.items():
                lines.append(f"  {qtype}:")
                for name, value in sorted(metrics.items()):
                    lines.append(f"    {name}: {value:.4f}")
        
        if self.reranker_improvement:
            lines.append("")
            lines.append("Reranker Improvement:")
            for name, value in sorted(self.reranker_improvement.items()):
                lines.append(f"  {name}: {value:+.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class RAGEvaluator:
    """Evaluator for RAG retrieval quality."""
    
    def __init__(
        self,
        retriever: GameRetriever,
        k_values: list[int] = None,
    ):
        """Initialize the evaluator.
        
        Args:
            retriever: GameRetriever instance to evaluate
            k_values: List of K values to compute metrics for
        """
        self.retriever = retriever
        self.k_values = k_values or [3, 5, 10]
    
    def evaluate_single(
        self,
        query: EvalQuery,
        k: int = 5,
        use_reranker: bool = True,
    ) -> EvaluationResult:
        """Evaluate a single query.
        
        Args:
            query: EvalQuery to evaluate
            k: Top-K for retrieval
            use_reranker: Whether to use reranker
            
        Returns:
            EvaluationResult with metrics
        """
        start_time = time.perf_counter()
        
        # Retrieve
        result = self.retriever.retrieve(
            query=query.query,
            top_k=k,
            use_reranker=use_reranker,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract retrieved game names
        retrieved_games = [
            chunk.get("metadata", {}).get("game_name", "")
            for chunk in result.chunks
        ]
        # Remove duplicates while preserving order
        seen = set()
        unique_retrieved = []
        for game in retrieved_games:
            if game and game not in seen:
                seen.add(game)
                unique_retrieved.append(game)
        
        # Compute metrics
        relevant_set = set(query.relevant_games)
        metrics = RetrievalMetrics.compute_all_metrics(
            retrieved=unique_retrieved,
            relevant=relevant_set,
            k=k,
        )
        
        return EvaluationResult(
            query=query.query,
            query_type=query.query_type,
            retrieved_games=unique_retrieved,
            relevant_games=query.relevant_games,
            metrics=metrics,
            latency_ms=latency_ms,
            reranked=result.reranked,
        )
    
    def evaluate_dataset(
        self,
        queries: list[EvalQuery],
        k: int = 5,
        use_reranker: bool = True,
        show_progress: bool = True,
    ) -> EvaluationReport:
        """Evaluate a full dataset.
        
        Args:
            queries: List of EvalQuery to evaluate
            k: Top-K for retrieval
            use_reranker: Whether to use reranker
            show_progress: Whether to show progress bar
            
        Returns:
            EvaluationReport with aggregated metrics
        """
        results = []
        metrics_by_type = {}
        
        iterator = tqdm(queries, desc="Evaluating") if show_progress else queries
        
        for query in iterator:
            result = self.evaluate_single(query, k=k, use_reranker=use_reranker)
            results.append(result)
            
            # Group by query type
            qtype = result.query_type
            if qtype not in metrics_by_type:
                metrics_by_type[qtype] = []
            metrics_by_type[qtype].append(result.metrics)
        
        # Aggregate metrics
        all_metrics = [r.metrics for r in results]
        aggregated = RetrievalMetrics.aggregate_metrics(all_metrics)
        
        # Aggregate by type
        type_aggregated = {}
        for qtype, type_metrics in metrics_by_type.items():
            type_aggregated[qtype] = RetrievalMetrics.aggregate_metrics(type_metrics)
        
        # Average latency
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
        
        return EvaluationReport(
            total_queries=len(queries),
            aggregated_metrics=aggregated,
            metrics_by_type=type_aggregated,
            avg_latency_ms=avg_latency,
            individual_results=results,
        )
    
    def compare_with_without_reranker(
        self,
        queries: list[EvalQuery],
        k: int = 5,
        show_progress: bool = True,
    ) -> tuple[EvaluationReport, EvaluationReport, dict[str, float]]:
        """Compare retrieval with and without reranker.
        
        Args:
            queries: List of EvalQuery to evaluate
            k: Top-K for retrieval
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (without_reranker_report, with_reranker_report, improvement)
        """
        print("Evaluating WITHOUT reranker...")
        report_without = self.evaluate_dataset(
            queries, k=k, use_reranker=False, show_progress=show_progress
        )
        
        print("Evaluating WITH reranker...")
        report_with = self.evaluate_dataset(
            queries, k=k, use_reranker=True, show_progress=show_progress
        )
        
        # Calculate improvement
        improvement = {}
        for metric_name in report_without.aggregated_metrics:
            without_val = report_without.aggregated_metrics[metric_name]
            with_val = report_with.aggregated_metrics.get(metric_name, 0)
            improvement[metric_name] = with_val - without_val
        
        # Add improvement to with report
        report_with.reranker_improvement = improvement
        
        return report_without, report_with, improvement
    
    def save_report(
        self,
        report: EvaluationReport,
        path: Path,
    ):
        """Save evaluation report to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    def evaluate_at_multiple_k(
        self,
        queries: list[EvalQuery],
        use_reranker: bool = True,
        show_progress: bool = True,
    ) -> dict[int, EvaluationReport]:
        """Evaluate at multiple K values.
        
        Args:
            queries: List of EvalQuery
            use_reranker: Whether to use reranker
            show_progress: Whether to show progress bar
            
        Returns:
            Dict mapping K value to EvaluationReport
        """
        reports = {}
        
        for k in self.k_values:
            print(f"Evaluating at K={k}...")
            report = self.evaluate_dataset(
                queries, k=k, use_reranker=use_reranker, show_progress=show_progress
            )
            reports[k] = report
        
        return reports


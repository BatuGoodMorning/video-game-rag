"""Retrieval evaluation metrics for RAG systems.

Implements standard IR metrics:
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
"""

import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    value: float
    k: int
    name: str
    
    def __repr__(self) -> str:
        return f"{self.name}@{self.k}: {self.value:.4f}"


class RetrievalMetrics:
    """Collection of retrieval evaluation metrics."""
    
    @staticmethod
    def precision_at_k(
        retrieved: list[str],
        relevant: set[str],
        k: int,
    ) -> float:
        """Calculate Precision@K.
        
        Precision@K = (# of relevant items in top-K) / K
        
        Args:
            retrieved: List of retrieved item IDs (in order)
            relevant: Set of relevant item IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K value between 0 and 1
        """
        if k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(
        retrieved: list[str],
        relevant: set[str],
        k: int,
    ) -> float:
        """Calculate Recall@K.
        
        Recall@K = (# of relevant items in top-K) / (total # of relevant items)
        
        Args:
            retrieved: List of retrieved item IDs (in order)
            relevant: Set of relevant item IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K value between 0 and 1
        """
        if not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant)
        
        return relevant_in_top_k / len(relevant)
    
    @staticmethod
    def hit_rate_at_k(
        retrieved: list[str],
        relevant: set[str],
        k: int,
    ) -> float:
        """Calculate Hit Rate@K (binary: 1 if any relevant item in top-K, else 0).
        
        Args:
            retrieved: List of retrieved item IDs (in order)
            relevant: Set of relevant item IDs
            k: Number of top results to consider
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = retrieved[:k]
        return 1.0 if any(item in relevant for item in top_k) else 0.0
    
    @staticmethod
    def reciprocal_rank(
        retrieved: list[str],
        relevant: set[str],
    ) -> float:
        """Calculate Reciprocal Rank.
        
        RR = 1 / (rank of first relevant item)
        
        Args:
            retrieved: List of retrieved item IDs (in order)
            relevant: Set of relevant item IDs
            
        Returns:
            Reciprocal rank value (0 if no relevant item found)
        """
        for i, item in enumerate(retrieved, 1):
            if item in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def mrr(
        queries_results: list[tuple[list[str], set[str]]],
    ) -> float:
        """Calculate Mean Reciprocal Rank over multiple queries.
        
        MRR = (1/|Q|) * sum(1/rank_i for i in Q)
        
        Args:
            queries_results: List of (retrieved, relevant) tuples for each query
            
        Returns:
            MRR value
        """
        if not queries_results:
            return 0.0
        
        rr_sum = sum(
            RetrievalMetrics.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in queries_results
        )
        
        return rr_sum / len(queries_results)
    
    @staticmethod
    def dcg_at_k(
        retrieved: list[str],
        relevance_scores: dict[str, float],
        k: int,
    ) -> float:
        """Calculate Discounted Cumulative Gain at K.
        
        DCG@K = sum(rel_i / log2(i + 1) for i in 1..K)
        
        Args:
            retrieved: List of retrieved item IDs (in order)
            relevance_scores: Dict mapping item ID to relevance score
            k: Number of top results to consider
            
        Returns:
            DCG@K value
        """
        dcg = 0.0
        for i, item in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(item, 0.0)
            dcg += rel / math.log2(i + 1)
        return dcg
    
    @staticmethod
    def ndcg_at_k(
        retrieved: list[str],
        relevance_scores: dict[str, float],
        k: int,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG@K = DCG@K / IDCG@K
        
        Where IDCG@K is the ideal DCG (if results were perfectly ranked).
        
        Args:
            retrieved: List of retrieved item IDs (in order)
            relevance_scores: Dict mapping item ID to relevance score
            k: Number of top results to consider
            
        Returns:
            NDCG@K value between 0 and 1
        """
        # Calculate actual DCG
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevance_scores, k)
        
        # Calculate ideal DCG (perfect ranking)
        ideal_order = sorted(
            relevance_scores.keys(),
            key=lambda x: relevance_scores.get(x, 0),
            reverse=True
        )
        idcg = RetrievalMetrics.dcg_at_k(ideal_order, relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def compute_all_metrics(
        retrieved: list[str],
        relevant: set[str],
        k: int = 5,
        relevance_scores: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Compute all metrics for a single query.
        
        Args:
            retrieved: List of retrieved item IDs (in order)
            relevant: Set of relevant item IDs
            k: Number of top results to consider
            relevance_scores: Optional graded relevance scores for NDCG
            
        Returns:
            Dict with all metric values
        """
        # Use binary relevance if no graded scores provided
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}
        
        return {
            f"precision@{k}": RetrievalMetrics.precision_at_k(retrieved, relevant, k),
            f"recall@{k}": RetrievalMetrics.recall_at_k(retrieved, relevant, k),
            f"hit_rate@{k}": RetrievalMetrics.hit_rate_at_k(retrieved, relevant, k),
            "mrr": RetrievalMetrics.reciprocal_rank(retrieved, relevant),
            f"ndcg@{k}": RetrievalMetrics.ndcg_at_k(retrieved, relevance_scores, k),
        }
    
    @staticmethod
    def aggregate_metrics(
        all_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate metrics across multiple queries.
        
        Args:
            all_metrics: List of metric dicts from compute_all_metrics
            
        Returns:
            Dict with averaged metrics
        """
        if not all_metrics:
            return {}
        
        aggregated = {}
        metric_names = all_metrics[0].keys()
        
        for name in metric_names:
            values = [m[name] for m in all_metrics]
            aggregated[f"avg_{name}"] = sum(values) / len(values)
        
        return aggregated


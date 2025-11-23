"""Evaluation metrics for RAG system"""
import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_recall_at_k(
        retrieved_docs: List[str],
        ground_truth_doc: str,
        k: int
) -> bool:
    """
    Calculate if ground truth document is in top-k retrieved documents

    Args:
        retrieved_docs: List of retrieved document names
        ground_truth_doc: Expected document name
        k: Number of top documents to consider

    Returns:
        True if ground truth in top-k, False otherwise
    """
    top_k_docs = retrieved_docs[:k]
    return ground_truth_doc in top_k_docs


def calculate_mrr(
        retrieved_docs: List[str],
        ground_truth_doc: str
) -> float:
    """
    Calculate Mean Reciprocal Rank

    Args:
        retrieved_docs: List of retrieved document names
        ground_truth_doc: Expected document name

    Returns:
        Reciprocal rank (1/position) or 0 if not found
    """
    try:
        position = retrieved_docs.index(ground_truth_doc) + 1  # 1-indexed
        return 1.0 / position
    except ValueError:
        return 0.0


class RAGEvaluator:
    """Evaluate RAG system performance"""

    def __init__(self, golden_set_path: str):
        """
        Initialize evaluator with golden set

        Args:
            golden_set_path: Path to golden set JSON file
        """
        import json

        self.golden_set_path = Path(golden_set_path)

        # Load golden set
        with open(self.golden_set_path, 'r', encoding='utf-8') as f:
            self.golden_set = json.load(f)

        logger.info(f"Loaded {len(self.golden_set)} Q&A pairs from {golden_set_path}")

    def evaluate(
            self,
            retriever,
            top_k: int = 5,
            sample_size: int = None
    ) -> Dict:
        """
        Evaluate retrieval performance

        Args:
            retriever: DocumentRetriever instance
            top_k: Number of documents to retrieve
            sample_size: Number of samples to evaluate (None = all)

        Returns:
            Dict with metrics
        """
        import time

        # Sample golden set if needed
        samples = self.golden_set[:sample_size] if sample_size else self.golden_set
        logger.info(f"Evaluating on {len(samples)} samples with top_k={top_k}")

        # Metrics
        recall_at_3 = []
        recall_at_5 = []
        mrr_scores = []
        latencies = []

        for idx, item in enumerate(samples):
            query = item["question"]
            ground_truth_doc = item["document"]

            # Retrieve
            start_time = time.time()
            chunks = retriever.retrieve(query, top_k=top_k)
            latency = time.time() - start_time

            # Extract document names
            retrieved_docs = [
                chunk["source"].split("/")[-1]  # Extract filename
                for chunk in chunks
            ]

            # Calculate metrics
            recall_at_3.append(calculate_recall_at_k(retrieved_docs, ground_truth_doc, k=3))
            recall_at_5.append(calculate_recall_at_k(retrieved_docs, ground_truth_doc, k=5))
            mrr_scores.append(calculate_mrr(retrieved_docs, ground_truth_doc))
            latencies.append(latency)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(samples)} samples")

        # Aggregate metrics
        results = {
            "recall@3": sum(recall_at_3) / len(recall_at_3),
            "recall@5": sum(recall_at_5) / len(recall_at_5),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
            "total_samples": len(samples),
            "top_k": top_k
        }

        logger.info(f"Evaluation complete: {results}")
        return results
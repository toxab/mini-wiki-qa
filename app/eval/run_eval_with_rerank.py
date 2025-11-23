"""Run evaluation with reranking"""
import logging
import mlflow
from pathlib import Path

from eval.metrics import RAGEvaluator
from rag.retrieval import get_retriever
from rag.reranker import get_reranker
from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGEvaluatorWithRerank(RAGEvaluator):
    """Evaluator with reranking support"""

    def evaluate(
            self,
            retriever,
            top_k: int = 5,
            sample_size: int = None,
            use_rerank: bool = False,
            reranker=None
    ):
        """
        Evaluate retrieval with optional reranking

        Args:
            retriever: DocumentRetriever instance
            top_k: Number of documents to retrieve
            sample_size: Number of samples (None = all)
            use_rerank: Whether to use reranking
            reranker: Reranker instance (required if use_rerank=True)
        """
        import time

        samples = self.golden_set[:sample_size] if sample_size else self.golden_set
        logger.info(f"Evaluating on {len(samples)} samples with top_k={top_k}, rerank={use_rerank}")

        recall_at_3 = []
        recall_at_5 = []
        mrr_scores = []
        latencies = []

        for idx, item in enumerate(samples):
            query = item["question"]
            ground_truth_doc = item["document"]

            # Retrieve
            start_time = time.time()
            if use_rerank:
                # Retrieve more candidates for reranking
                chunks = retriever.retrieve(query, top_k=20)
            else:
                chunks = retriever.retrieve(query, top_k=top_k)

            # Rerank if enabled
            if use_rerank and reranker:
                chunks = reranker.rerank(query, chunks, top_k=top_k)

            latency = time.time() - start_time

            # Extract document names
            retrieved_docs = [
                chunk["source"].split("/")[-1]
                for chunk in chunks
            ]

            # Calculate metrics
            from eval.metrics import calculate_recall_at_k, calculate_mrr
            recall_at_3.append(calculate_recall_at_k(retrieved_docs, ground_truth_doc, k=3))
            recall_at_5.append(calculate_recall_at_k(retrieved_docs, ground_truth_doc, k=5))
            mrr_scores.append(calculate_mrr(retrieved_docs, ground_truth_doc))
            latencies.append(latency)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(samples)} samples")

        results = {
            "recall@3": sum(recall_at_3) / len(recall_at_3),
            "recall@5": sum(recall_at_5) / len(recall_at_5),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
            "total_samples": len(samples),
            "top_k": top_k,
            "use_rerank": use_rerank
        }

        logger.info(f"Evaluation complete: {results}")
        return results


def run_evaluation_with_rerank(
        golden_set_path: str = "/data/golden_set/squad_qa.json",
        top_k: int = 5,
        sample_size: int = 50,
        use_rerank: bool = True,
        experiment_name: str = "rag-with-rerank"
):
    """Run evaluation with reranking and log to MLflow"""
    logger.info("🚀 Starting evaluation with reranking...")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Initialize
    retriever = get_retriever()
    reranker = get_reranker() if use_rerank else None
    evaluator = RAGEvaluatorWithRerank(golden_set_path)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("chunk_size", settings.CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", settings.CHUNK_OVERLAP)
        mlflow.log_param("embedding_model", settings.EMBEDDING_MODEL)
        mlflow.log_param("use_rerank", use_rerank)
        mlflow.log_param("sample_size", sample_size)

        # Run evaluation
        results = evaluator.evaluate(
            retriever=retriever,
            top_k=top_k,
            sample_size=sample_size,
            use_rerank=use_rerank,
            reranker=reranker
        )

        # Log metrics
        mlflow.log_metric("recall_at_3", results["recall@3"])
        mlflow.log_metric("recall_at_5", results["recall@5"])
        mlflow.log_metric("mrr", results["mrr"])
        mlflow.log_metric("avg_latency_ms", results["avg_latency_ms"])

        logger.info("✅ Evaluation complete!")
        logger.info(f"📊 Results:")
        logger.info(f"  Recall@3: {results['recall@3']:.3f}")
        logger.info(f"  Recall@5: {results['recall@5']:.3f}")
        logger.info(f"  MRR: {results['mrr']:.3f}")
        logger.info(f"  Avg Latency: {results['avg_latency_ms']:.1f}ms")

        return results


if __name__ == "__main__":
    # Run with reranking
    run_evaluation_with_rerank(
        top_k=5,
        sample_size=50,
        use_rerank=True,
        experiment_name="rag-with-rerank"
    )
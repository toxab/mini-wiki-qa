"""Run evaluation with MLflow tracking"""
import logging
import mlflow
from pathlib import Path

from eval.metrics import RAGEvaluator
from rag.retrieval import get_retriever
from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_evaluation(
        golden_set_path: str = "/data/golden_set/squad_qa.json",
        top_k: int = 5,
        sample_size: int = None,
        experiment_name: str = "rag-baseline"
):
    """
    Run evaluation and log to MLflow

    Args:
        golden_set_path: Path to golden set
        top_k: Number of documents to retrieve
        sample_size: Number of samples (None = all)
        experiment_name: MLflow experiment name
    """
    logger.info("🚀 Starting evaluation...")

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Initialize components
    logger.info("Initializing retriever...")
    retriever = get_retriever()

    logger.info("Loading golden set...")
    evaluator = RAGEvaluator(golden_set_path)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("chunk_size", settings.CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", settings.CHUNK_OVERLAP)
        mlflow.log_param("embedding_model", settings.EMBEDDING_MODEL)
        mlflow.log_param("sample_size", sample_size or len(evaluator.golden_set))

        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluator.evaluate(
            retriever=retriever,
            top_k=top_k,
            sample_size=sample_size
        )

        # Log metrics
        mlflow.log_metric("recall_at_3", results["recall@3"])
        mlflow.log_metric("recall_at_5", results["recall@5"])
        mlflow.log_metric("mrr", results["mrr"])
        mlflow.log_metric("avg_latency_ms", results["avg_latency_ms"])

        # Log results as artifact
        results_path = Path("/tmp/eval_results.json")
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(results_path))

        logger.info("✅ Evaluation complete!")
        logger.info(f"📊 Results:")
        logger.info(f"  Recall@3: {results['recall@3']:.3f}")
        logger.info(f"  Recall@5: {results['recall@5']:.3f}")
        logger.info(f"  MRR: {results['mrr']:.3f}")
        logger.info(f"  Avg Latency: {results['avg_latency_ms']:.1f}ms")

        return results


if __name__ == "__main__":
    # Run baseline evaluation
    run_evaluation(
        top_k=5,
        sample_size=50,  # Evaluate on 50 samples for speed
        experiment_name="rag-baseline"
    )
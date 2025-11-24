"""Test MLflow + MinIO integration"""
import sys

sys.path.insert(0, '/app')

import mlflow
import os

# MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")


def test_mlflow_minio():
    """Test artifact logging to MinIO"""

    print("🧪 Testing MLflow + MinIO integration...")

    # Set experiment
    experiment_name = "minio-test"
    mlflow.set_experiment(experiment_name)

    # Start run
    with mlflow.start_run(run_name="test-artifacts"):
        # Log parameters
        mlflow.log_param("test_param", "minio_integration")

        # Log metrics
        mlflow.log_metric("test_metric", 42.0)

        # Create test file
        test_file = "/tmp/test_artifact.txt"
        with open(test_file, "w") as f:
            f.write("Hello from MinIO!\n")
            f.write("This artifact is stored in S3-compatible storage.\n")

        # Log artifact (should go to MinIO)
        mlflow.log_artifact(test_file, "test_artifacts")

        print("✅ Run logged successfully!")
        print(f"📊 Check MLflow UI: http://localhost:5001")
        print(f"🗄️  Check MinIO: http://localhost:9001/browser/mlflow-artifacts")

    print("✅ Test complete!")


if __name__ == "__main__":
    test_mlflow_minio()
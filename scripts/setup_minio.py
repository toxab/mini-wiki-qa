"""Setup MinIO buckets and configure storage"""
import sys

sys.path.insert(0, '/app')

from minio import Minio
from minio.error import S3Error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_minio():
    """Initialize MinIO buckets"""

    # MinIO client
    client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

    # Buckets to create
    buckets = [
        "mlflow-artifacts",
        "eval-results",
        "documents-backup"
    ]

    logger.info("🗄️  Setting up MinIO buckets...")

    for bucket_name in buckets:
        try:
            # Check if bucket exists
            if client.bucket_exists(bucket_name):
                logger.info(f"✅ Bucket '{bucket_name}' already exists")
            else:
                # Create bucket
                client.make_bucket(bucket_name)
                logger.info(f"✅ Created bucket: {bucket_name}")

        except S3Error as e:
            logger.error(f"❌ Error with bucket '{bucket_name}': {e}")

    logger.info("✅ MinIO setup complete!")
    logger.info("📊 Available buckets:")

    # List all buckets
    buckets_list = client.list_buckets()
    for bucket in buckets_list:
        logger.info(f"  • {bucket.name}")


if __name__ == "__main__":
    setup_minio()
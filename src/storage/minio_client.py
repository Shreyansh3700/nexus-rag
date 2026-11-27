"""MinIO client for document storage.

Provides upload, download, and delete operations for raw uploaded files.
The worker downloads files from MinIO during ingestion, then the original
file remains available for re-processing or auditing.
"""
import io
from functools import lru_cache

from minio import Minio
from minio.error import S3Error

from src.config.settings import settings
from src.logger import get_logger

logger = get_logger(__name__)


class MinioClient:
    """Wrapper around MinIO SDK for document storage operations."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ):
        """Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint (e.g., "localhost:9000")
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Whether to use HTTPS (default False for local dev)
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        logger.debug("MinIO client initialized: endpoint=%s secure=%s", endpoint, secure)

    def ensure_bucket(self, bucket: str) -> None:
        """Create bucket if it doesn't exist.

        Args:
            bucket: Bucket name

        Raises:
            S3Error: If bucket creation fails
        """
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info("Created MinIO bucket: %s", bucket)
            else:
                logger.debug("MinIO bucket already exists: %s", bucket)
        except S3Error as exc:
            logger.exception("Failed to create MinIO bucket: %s", bucket)
            raise RuntimeError(f"MinIO bucket creation failed: {exc}") from exc

    def upload_file(self, bucket: str, object_name: str, data: bytes) -> str:
        """Upload file bytes to MinIO.

        The bucket must already exist: the API ensures it once at startup
        (see set_environment in src/config/config.py).

        Args:
            bucket: Bucket name
            object_name: Object key (e.g., "document_id.pdf")
            data: File bytes to upload

        Returns:
            Object URL (for logging/debugging, not for public access)

        Raises:
            RuntimeError: If upload fails
        """
        try:
            data_stream = io.BytesIO(data)
            self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=data_stream,
                length=len(data),
            )
            logger.info(
                "Uploaded file to MinIO: bucket=%s object=%s size=%s",
                bucket,
                object_name,
                len(data),
            )
            return f"{bucket}/{object_name}"
        except S3Error as exc:
            logger.exception(
                "Failed to upload file to MinIO: bucket=%s object=%s",
                bucket,
                object_name,
            )
            raise RuntimeError(f"MinIO upload failed: {exc}") from exc

    def download_file(self, bucket: str, object_name: str) -> bytes:
        """Download file bytes from MinIO.

        Args:
            bucket: Bucket name
            object_name: Object key

        Returns:
            File bytes

        Raises:
            RuntimeError: If download fails
        """
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            logger.info(
                "Downloaded file from MinIO: bucket=%s object=%s size=%s",
                bucket,
                object_name,
                len(data),
            )
            return data
        except S3Error as exc:
            logger.exception(
                "Failed to download file from MinIO: bucket=%s object=%s",
                bucket,
                object_name,
            )
            raise RuntimeError(f"MinIO download failed: {exc}") from exc

    def delete_file(self, bucket: str, object_name: str) -> None:
        """Delete file from MinIO.

        Args:
            bucket: Bucket name
            object_name: Object key

        Raises:
            RuntimeError: If delete fails
        """
        try:
            self.client.remove_object(bucket, object_name)
            logger.info(
                "Deleted file from MinIO: bucket=%s object=%s",
                bucket,
                object_name,
            )
        except S3Error as exc:
            logger.exception(
                "Failed to delete file from MinIO: bucket=%s object=%s",
                bucket,
                object_name,
            )
            raise RuntimeError(f"MinIO delete failed: {exc}") from exc


@lru_cache(maxsize=1)
def get_minio_client() -> MinioClient:
    """Get or create the singleton MinIO client (MINIO_* settings)."""
    logger.debug("Creating MinIO client singleton: endpoint=%s", settings.minio_endpoint)
    return MinioClient(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key.get_secret_value(),
        secure=settings.minio_secure,
    )

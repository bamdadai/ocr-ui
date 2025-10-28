"""
Hybrid Storage Manager for OCR Pipeline

Intelligent routing:
- Files < 50MB: Store in Redis (faster access)
- Files >= 50MB: Store in MinIO (preserve memory)

This maximizes performance while preventing OOM on large files.
"""

from typing import Optional, Dict, Tuple
import structlog
from minio import Minio
from minio.error import S3Error
import io

logger = structlog.get_logger(__name__)

# MinIO threshold: 50MB
MINIO_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50MB


class HybridStorageManager:
    """
    Smart storage routing based on file size.
    
    Strategy:
    - Small files (<50MB) → Redis (fast, in-memory)
    - Large files (≥50MB) → MinIO (scalable, preserves RAM)
    
    This provides best of both worlds:
    - Speed for typical files
    - Memory efficiency for large files
    """
    
    BUCKET_NAME = "ocr-staging"
    REGION_NAME = "us-east-1"
    
    def __init__(self, threshold_mb: int = 50, endpoint: str = "minio:9000",
                 access_key: str = "minioadmin",
                 secret_key: str = "minioadmin2025"):
        """Initialize hybrid storage with MinIO client.
        
        Args:
            threshold_mb: File size threshold in MB (default: 50)
            endpoint: MinIO endpoint (default: minio:9000)
            access_key: MinIO access key
            secret_key: MinIO secret key
        """
        self.threshold_mb = threshold_mb
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.endpoint = endpoint
        
        try:
            self.minio_client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False
            )
            # Ensure bucket exists
            if not self.minio_client.bucket_exists(self.BUCKET_NAME):
                self.minio_client.make_bucket(self.BUCKET_NAME, region=self.REGION_NAME)
            
            logger.info("hybrid_storage.initialized", endpoint=endpoint, threshold_mb=threshold_mb)
        except Exception as e:
            logger.error("hybrid_storage.minio_init_failed", error=str(e))
            raise
    
    def get_storage_type(self, file_size: int) -> str:
        """
        Determine storage type based on file size.
        
        Args:
            file_size: File size in bytes
            
        Returns: "redis" or "minio"
        """
        if file_size >= self.threshold_bytes:
            return "minio"
        return "redis"
    
    def upload_to_minio(self, request_id: str, file_bytes: bytes, 
                       filename: str) -> Dict:
        """Upload large file to MinIO."""
        object_name = f"{request_id}/{filename}"
        file_size = len(file_bytes)
        
        try:
            self.minio_client.put_object(
                bucket_name=self.BUCKET_NAME,
                object_name=object_name,
                data=io.BytesIO(file_bytes),
                length=file_size
            )
            
            logger.info(
                "hybrid_storage.uploaded_to_minio",
                request_id=request_id,
                filename=filename,
                size_mb=file_size / (1024 * 1024),
                storage="minio"
            )
            
            return {
                "request_id": request_id,
                "object_name": object_name,
                "filename": filename,
                "size_bytes": file_size,
                "storage": "minio"
            }
        except S3Error as e:
            logger.error("hybrid_storage.minio_upload_failed", error=str(e))
            raise
    
    def download_from_minio(self, request_id: str, filename: str) -> bytes:
        """Download large file from MinIO."""
        object_name = f"{request_id}/{filename}"
        
        try:
            response = self.minio_client.get_object(
                bucket_name=self.BUCKET_NAME,
                object_name=object_name
            )
            file_bytes = response.read()
            
            logger.debug(
                "hybrid_storage.downloaded_from_minio",
                request_id=request_id,
                filename=filename,
                size_mb=len(file_bytes) / (1024 * 1024)
            )
            
            return file_bytes
        except S3Error as e:
            logger.error("hybrid_storage.minio_download_failed", error=str(e))
            raise
    
    def delete_from_minio(self, request_id: str, filename: str) -> bool:
        """Delete file from MinIO."""
        object_name = f"{request_id}/{filename}"
        
        try:
            self.minio_client.remove_object(
                bucket_name=self.BUCKET_NAME,
                object_name=object_name
            )
            logger.debug("hybrid_storage.deleted_from_minio", 
                        request_id=request_id, filename=filename)
            return True
        except S3Error:
            logger.warning("hybrid_storage.minio_delete_failed",
                          request_id=request_id, filename=filename)
            return False
    
    def cleanup_minio_request(self, request_id: str) -> int:
        """Delete all files for a request from MinIO."""
        try:
            objects = self.minio_client.list_objects(
                bucket_name=self.BUCKET_NAME,
                prefix=f"{request_id}/"
            )
            
            deleted_count = 0
            for obj in objects:
                try:
                    self.minio_client.remove_object(
                        bucket_name=self.BUCKET_NAME,
                        object_name=obj.object_name
                    )
                    deleted_count += 1
                except S3Error:
                    continue
            
            logger.info("hybrid_storage.minio_cleanup", 
                       request_id=request_id, deleted_count=deleted_count)
            return deleted_count
        except S3Error as e:
            logger.error("hybrid_storage.minio_cleanup_failed", error=str(e))
            return 0


# Singleton instance
_hybrid_storage: Optional[HybridStorageManager] = None


def get_hybrid_storage() -> HybridStorageManager:
    """Get or create hybrid storage manager (singleton)."""
    global _hybrid_storage
    
    if _hybrid_storage is None:
        _hybrid_storage = HybridStorageManager()
    
    return _hybrid_storage

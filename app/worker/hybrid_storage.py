"""
Hybrid Storage Manager for OCR Pipeline

Intelligent routing:
- Files < threshold: Store in Redis (faster access)
- Files >= threshold: Store in MinIO (preserve memory)

This maximizes performance while preventing OOM on large files.
"""

from typing import Optional, Dict
import structlog
try:
    from minio import Minio
    from minio.error import S3Error
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False
    import warnings
    warnings.warn("minio library not available. Install with: pip install minio")

import io

logger = structlog.get_logger(__name__)


class HybridStorageManager:
    """
    Smart storage routing based on file size.
    
    Strategy:
    - Small files (<threshold) → Redis (fast, in-memory)
    - Large files (>=threshold) → MinIO (scalable, preserves RAM)
    
    This provides best of both worlds:
    - Speed for typical files
    - Memory efficiency for large files
    """
    
    def __init__(self, config):
        """Initialize hybrid storage with MinIO client from config.
        
        Args:
            config: HybridStorageConfig from settings
        """
        self.config = config
        self.threshold_mb = config.threshold_mb
        self.threshold_bytes = config.threshold_mb * 1024 * 1024
        self.enabled = config.enabled
        
        if not self.enabled:
            logger.info("hybrid_storage.disabled", reason="Config disabled")
            self.minio_client = None
            return
        
        if not HAS_MINIO:
            logger.error("hybrid_storage.minio_not_available", reason="Library not installed")
            raise RuntimeError("MinIO library not available. Install with: pip install minio")
        
        minio_config = config.minio
        
        try:
            self.minio_client = Minio(
                minio_config.endpoint,
                access_key=minio_config.access_key,
                secret_key=minio_config.secret_key,
                secure=minio_config.secure
            )
            # Ensure bucket exists
            if not self.minio_client.bucket_exists(minio_config.bucket_name):
                # MinIO make_bucket() doesn't accept region parameter
                self.minio_client.make_bucket(minio_config.bucket_name)
            
            logger.info(
                "hybrid_storage.initialized", 
                endpoint=minio_config.endpoint, 
                threshold_mb=self.threshold_mb,
                bucket=minio_config.bucket_name
            )
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
        if not self.enabled or self.minio_client is None:
            raise RuntimeError("Hybrid storage is not enabled or MinIO client not initialized")
        
        minio_config = self.config.minio
        object_name = f"{request_id}/{filename}"
        file_size = len(file_bytes)
        
        try:
            self.minio_client.put_object(
                bucket_name=minio_config.bucket_name,
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
        if not self.enabled or self.minio_client is None:
            raise RuntimeError("Hybrid storage is not enabled or MinIO client not initialized")
        
        minio_config = self.config.minio
        object_name = f"{request_id}/{filename}"
        
        try:
            response = self.minio_client.get_object(
                bucket_name=minio_config.bucket_name,
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
        if not self.enabled or self.minio_client is None:
            return False
        
        minio_config = self.config.minio
        object_name = f"{request_id}/{filename}"
        
        try:
            self.minio_client.remove_object(
                bucket_name=minio_config.bucket_name,
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
        if not self.enabled or self.minio_client is None:
            return 0
        
        minio_config = self.config.minio
        try:
            objects = self.minio_client.list_objects(
                bucket_name=minio_config.bucket_name,
                prefix=f"{request_id}/"
            )
            
            deleted_count = 0
            for obj in objects:
                try:
                    self.minio_client.remove_object(
                        bucket_name=minio_config.bucket_name,
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

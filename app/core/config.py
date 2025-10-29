import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically calculate the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Nested Configuration Models for better structure and readability ---

class DetectionParallelConfig(BaseModel):
    """Settings for parallel processing in the detection service."""
    enabled: bool = False
    max_workers: int = 2
    max_batch_size: int = 4
    memory_limit_mb: int = 2048

class DetectionConfig(BaseModel):
    """Settings related to the Detection service."""
    device: str
    word_detect: Dict[str, Any]
    paddle_ocr: Dict[str, Any] = {}
    parallel_processing: DetectionParallelConfig
    debug: bool
    debug_word_path: Path = PROJECT_ROOT / "debug/word_detections"
    debug_line_path: Path = PROJECT_ROOT / "debug/line_detections"

class RecognitionConfig(BaseModel):
    """Settings related to the Recognition service."""
    device: str
    batch_size: int
    min_conf: float
    checkpoint: Path
    debug: bool
    paddle_rec: Dict[str, Any] = {}
    debug_word_polygons_path: Path = PROJECT_ROOT / "debug/word_polygons"
    debug_parts_path: Path = PROJECT_ROOT / "debug/parts"
    debug_word_crops_path: Path = PROJECT_ROOT / "debug/word_crops"

class OrientationConfig(BaseModel):
    """Settings for optional orientation (rotation) correction using a classification model."""
    enabled: bool = False
    path: Optional[str] = None  # path to YOLO-cls .pt
    device: Optional[str] = None  # if None, fallback to detection.device
    imgsz: int = 224

class TextRenderingConfig(BaseModel):
    """Settings for text rendering and line grouping."""
    same_row_separator: str = " "
    height_tolerance_pixels: int = 5

class PipelineConfig(BaseModel):
    """General settings for the OCR pipeline."""
    debug: bool
    enable_recognition: bool
    # Allow nesting orientation config under pipeline as well (optional)
    orientation: Optional[OrientationConfig] = None

class UploadPolicyConfig(BaseModel):
    """Defines a size-based override for upload staging behaviour."""
    min_size_bytes: int
    chunk_size_bytes: int
    ttl_seconds: int

class UploadStorageConfig(BaseModel):
    """Controls how large uploads are staged in Redis."""
    chunk_size_bytes: int = 5 * 1024 * 1024
    ttl_seconds: int = 7200
    policies: List[UploadPolicyConfig] = []

    @field_validator("policies")
    @classmethod
    def sort_policies(cls, v: List[UploadPolicyConfig]) -> List[UploadPolicyConfig]:
        """Ensure policies are evaluated from the largest threshold downwards."""
        if v:
            return sorted(v, key=lambda policy: policy.min_size_bytes, reverse=True)
        return v

class MinIOConfig(BaseModel):
    """Configuration for MinIO object storage."""
    enabled: bool
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    region: str
    secure: bool = False

class HybridStorageConfig(BaseModel):
    """
    Intelligent storage routing to balance performance and memory:
    - Small files (<threshold) → Redis (fast, in-memory)
    - Large files (>=threshold) → MinIO (scalable, preserves RAM)
    """
    enabled: bool
    threshold_mb: int
    small_files_storage: str = "redis"
    large_files_storage: str = "minio"
    minio: MinIOConfig

# --- Main Application Settings Class ---

class Settings(BaseSettings):
    """
    The main class for managing all application settings.
    It loads settings from a .env file as well as a master JSON file.
    """
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra variables found in the .env file
    )
    
    # Settings read from environment variables or the .env file
    APP_NAME: str = "Unified OCR Service"
    CELERY_BROKER_URL: str = "amqp://user:bambamai2025@rabbitmq:5672//"
    CELERY_BACKEND_URL: str = "redis://redis:6379/0"
    LOG_FILE_PATH: Path = PROJECT_ROOT / "logs/app.log"
    SEARCHABLE_PDF_FONT_PATH: Path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ALLOW_INSECURE_WEBHOOKS: bool = True

    # Structural settings that are loaded from the JSON config file
    debug: bool
    detection: DetectionConfig
    recognition: RecognitionConfig
    pipeline: PipelineConfig
    text_rendering: TextRenderingConfig
    # Support top-level orientation configuration as well (optional)
    orientation: Optional[OrientationConfig] = None
    upload_storage: UploadStorageConfig = UploadStorageConfig()
    hybrid_storage: Optional[HybridStorageConfig] = None
    valid_ocr_formats: List[str]

    @field_validator("detection", "recognition", "orientation", mode='before')
    @classmethod
    def resolve_paths_in_config(cls, v: Any) -> Any:
        """
        A Pydantic field validator that recursively finds all relative paths
        (like model checkpoints) in nested dictionaries and resolves them
        to absolute paths based on the project root.
        """
        if isinstance(v, dict):
            for key, value in v.items():
                # If the key is a path-related key, resolve it to an absolute path
                if key in ('path', 'checkpoint', 'model_path', 'config_path') and isinstance(value, str):
                    path = Path(value)
                    if not path.is_absolute():
                        v[key] = str(PROJECT_ROOT / path)
                # If the value is another dictionary, recurse into it
                elif isinstance(value, dict):
                    v[key] = cls.resolve_paths_in_config(value)
        return v

# --- Settings Loader Function and Singleton Instance ---

def load_settings() -> Settings:
    """
    Loads settings from the master_config.json file.
    The path to the config file is read from the MASTER_CONFIG_PATH environment variable.
    If the variable is not set, it falls back to a default path in the project root.
    """
    # Default path for the configuration file
    default_path = PROJECT_ROOT / "master_config.json"
    
    # Read path from environment variable or use the default
    config_path_str = os.getenv("MASTER_CONFIG_PATH", str(default_path))
    config_path = Path(config_path_str)

    # Check for the config file's existence and provide a clear error if not found
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found. "
            f"Looked for it at: {config_path}. "
            f"You can set the path using the 'MASTER_CONFIG_PATH' environment variable."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        json_config_data = json.load(f)
        
    return Settings(**json_config_data)

# Create a singleton instance of the settings to be used throughout the application
settings = load_settings()

# --- Post-load Initial Setup ---

def setup_directories():
    """
    Creates necessary application directories on application startup.
    """
    settings.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if settings.debug:
        settings.detection.debug_word_path.mkdir(parents=True, exist_ok=True)
        settings.detection.debug_line_path.mkdir(parents=True, exist_ok=True)
        settings.recognition.debug_word_polygons_path.mkdir(parents=True, exist_ok=True)
        settings.recognition.debug_parts_path.mkdir(parents=True, exist_ok=True)
        settings.recognition.debug_word_crops_path.mkdir(parents=True, exist_ok=True)

# Run the function to create directories when the module is imported
setup_directories()

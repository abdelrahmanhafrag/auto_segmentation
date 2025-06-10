"""
Configuration settings for PET Segmentation system.
Uses Pydantic for validation and environment variable support.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ImageFormat(str, Enum):
    """Supported image formats for output."""
    NIFTI = "nifti"
    NIFTI_GZ = "nifti_gz"
    NUMPY = "numpy"
    DICOM = "dicom"


class Settings(BaseSettings):
    """Configuration settings with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="PET Segmentation", description="Application name")
    debug: bool = Field(default=False, env="DEBUG", description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    
    # nnInteractive server settings
    nninteractive_url: str = Field(
        default="http://localhost:1527",
        env="NNINTERACTIVE_URL",
        description="nnInteractive server URL"
    )
    nninteractive_timeout: int = Field(
        default=120,
        env="NNINTERACTIVE_TIMEOUT",
        description="Request timeout in seconds"
    )
    nninteractive_retries: int = Field(
        default=3,
        env="NNINTERACTIVE_RETRIES",
        description="Number of retry attempts"
    )
    
    # Image processing settings
    default_percentile_threshold: int = Field(
        default=90,
        ge=50,
        le=99,
        description="Percentile threshold for bright region detection"
    )
    min_region_size: int = Field(
        default=100,
        gt=0,
        description="Minimum size of regions to consider"
    )
    max_regions_per_image: int = Field(
        default=10,
        gt=0,
        le=50,
        description="Maximum regions to process per image"
    )
    
    # Output settings
    output_format: ImageFormat = Field(
        default=ImageFormat.NIFTI_GZ,
        env="OUTPUT_FORMAT"
    )
    output_dir: Path = Field(
        default=Path("./outputs"),
        env="OUTPUT_DIR",
        description="Output directory for segmentations"
    )
    save_intermediate: bool = Field(
        default=False,
        env="SAVE_INTERMEDIATE",
        description="Save intermediate processing results"
    )
    
    # Batch processing settings
    batch_size: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Number of files to process in parallel"
    )
    max_workers: int = Field(
        default=4,
        gt=0,
        le=16,
        description="Maximum number of worker processes"
    )
    
    # Quality control settings
    min_quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to accept segmentation"
    )
    max_components: int = Field(
        default=3,
        gt=0,
        description="Maximum allowed disconnected components"
    )
    
    # Database settings (optional)
    database_url: Optional[str] = Field(
        default=None,
        env="DATABASE_URL",
        description="Database URL for metadata storage"
    )
    
    # Redis settings (for caching and job queues)
    redis_url: Optional[str] = Field(
        default=None,
        env="REDIS_URL",
        description="Redis URL for caching and job queues"
    )
    
    # Security settings
    api_key: Optional[str] = Field(
        default=None,
        env="API_KEY",
        description="API key for authentication"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS",
        description="Allowed hosts for API access"
    )
    
    # File system settings
    temp_dir: Path = Field(
        default=Path("/tmp/pet_segmentation"),
        env="TEMP_DIR"
    )
    max_file_size: int = Field(
        default=1024 * 1024 * 1024,  # 1GB
        description="Maximum file size in bytes"
    )
    
    @validator("output_dir", "temp_dir")
    def create_directories(cls, v):
        """Ensure directories exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("nninteractive_url")
    def validate_url(cls, v):
        """Ensure URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v.rstrip("/")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Custom configuration for different environments
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def create_config_file(config_path: str = "config.yaml") -> None:
    """Create a sample configuration file."""
    import yaml
    
    settings = Settings()
    config_dict = {
        # Core settings
        "nninteractive_url": settings.nninteractive_url,
        "debug": settings.debug,
        "log_level": settings.log_level.value,
        
        # Processing settings
        "default_percentile_threshold": settings.default_percentile_threshold,
        "min_region_size": settings.min_region_size,
        "max_regions_per_image": settings.max_regions_per_image,
        
        # Output settings
        "output_format": settings.output_format.value,
        "output_dir": str(settings.output_dir),
        
        # Quality control
        "min_quality_score": settings.min_quality_score,
        "max_components": settings.max_components,
        
        # Performance
        "batch_size": settings.batch_size,
        "max_workers": settings.max_workers,
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration file created: {config_path}")


if __name__ == "__main__":
    # Create sample config when run directly
    create_config_file() 
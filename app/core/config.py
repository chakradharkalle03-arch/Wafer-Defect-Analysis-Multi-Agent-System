"""
Configuration settings for the Wafer Defect Analysis System
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    hf_api_key: str = os.getenv("HF_API_KEY", "your_huggingface_api_key_here")  # Get from environment variable
    
    # Model Configuration
    model_cache_dir: str = "./models_cache"
    device: str = "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu"
    
    # Model Names (HuggingFace Inference API)
    # Using HuggingFace Inference API for all models (no local YOLO)
    hf_object_detection_model: str = "facebook/detr-resnet-50"  # Object detection via HF API
    hf_image_classification_model: str = "google/vit-base-patch16-224"  # Image classification via HF API
    sam_model: str = "facebook/sam-vit-base"  # For future segmentation features
    
    # Dataset Configuration (HuggingFace)
    wafer_dataset: str = "lslattery/wafer-defect-detection"  # Primary wafer defect dataset
    semiconductor_dataset: str = "sitloboi2012/semiconductor_scirepeval_v1"  # Additional dataset
    use_dataset_reference: bool = True  # Use dataset images for validation
    
    # Agent Configuration
    confidence_threshold: float = 0.5
    max_defects_per_image: int = 50
    
    # Classification Categories
    defect_categories: list = [
        "CMP_defects",
        "litho_hotspots",
        "pattern_bridging",
        "scratches",
        "particles",
        "pattern_defects",
        "etch_defects",
        "deposition_defects"
    ]
    
    # Process Steps
    process_steps: list = [
        "CMP",
        "Lithography",
        "Etch",
        "Deposition",
        "Implant",
        "Cleaning"
    ]
    
    # API Configuration
    backend_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    
    # File Upload
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow reading from environment variables
        extra = "ignore"


settings = Settings()


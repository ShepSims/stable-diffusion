import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Model Settings
    DEFAULT_MODEL: str = "runwayml/stable-diffusion-v1-5"
    MODELS_DIR: str = "models"
    UPLOADS_DIR: str = "uploads"
    OUTPUTS_DIR: str = "outputs"
    
    # Training Settings
    MAX_TRAINING_IMAGES: int = 25
    DEFAULT_RESOLUTION: int = 512
    DEFAULT_TRAINING_STEPS: int = 1000
    DEFAULT_LEARNING_RATE: float = 1e-4
    
    # AWS Settings
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = ""
    
    # Compute Settings
    DEVICE: str = "mps"  # Use Metal Performance Shaders on Mac
    MIXED_PRECISION: str = "no"  # MPS doesn't support fp16 mixed precision
    USE_XFORMERS: bool = False
    
    # Security
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_EXTENSIONS: list = [".png", ".jpg", ".jpeg", ".webp"]
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create directories
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
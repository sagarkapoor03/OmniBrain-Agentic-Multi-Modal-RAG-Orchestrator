import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Google AI Studio API Key
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"

    # Storage Paths
    QDRANT_PATH: str = str(BASE_DIR / "data" / "qdrant_db")
    QDRANT_COLLECTION_NAME: str = "nvda_documents"
    
    # SQLite Stock Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'stock_data' / 'stocks.db'}"
    STOCK_DB_PATH: str = str(BASE_DIR / "data" / "stock_data" / "stocks.db")
    NVDA_CSV_PATH: str = str(BASE_DIR / "data" / "stock_data" / "NVDA.csv")

    # Document & Image Folders
    DOCUMENTS_DIR: str = str(BASE_DIR / "data" / "documents")
    EXTRACTED_IMAGES_DIR: str = str(BASE_DIR / "data" / "documents" / "extracted_images")

    # Observability
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

settings = Settings()

# Ensure required directories exist
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(settings.EXTRACTED_IMAGES_DIR, exist_ok=True)
os.makedirs(Path(settings.STOCK_DB_PATH).parent, exist_ok=True)

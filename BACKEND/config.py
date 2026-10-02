import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    BASE_DIR = BASE_DIR
    MODELS_DIR = BASE_DIR / "models"

    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
    TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN", "").strip()

    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:5500")
    CORS_ORIGINS = [
        FRONTEND_URL,
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


settings = Settings()

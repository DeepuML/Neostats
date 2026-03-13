"""Centralized application configuration loaded from environment variables."""

from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    """Runtime settings for the chatbot application."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    search_api_key: str = os.getenv("SEARCH_API_KEY", "")

    openai_model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

def validate_api_key(provider_name: str) -> None:
    """Validating configured API key."""

    provider = provider_name.strip().lower()
    settings = get_settings()
    key_by_provider = {
        "openai": settings.openai_api_key,
        "groq": settings.groq_api_key,
        "gemini": settings.gemini_api_key,
    }

    if provider not in key_by_provider:
        raise ValueError(f"Unsupported provider: {provider_name}")

    if not key_by_provider[provider]:
        env_var_name = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[provider]
        raise ValueError(
            f"Missing API key for {provider_name}. Set environment variable {env_var_name}."
        )

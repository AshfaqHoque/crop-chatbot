"""
Central application settings.

Everything the pipeline needs that could change between environments
(model name, API URLs, timeouts, confidence thresholds) lives here and
ONLY here. No other module should read os.environ directly — this is
what "no hardcoding" means in practice: every tunable value has one
authoritative source, and swapping it (e.g. Gemma3 -> Qwen2.5, or
in-memory -> Redis later) means changing this file, not hunting through
the codebase.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama / LLM ---
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="gemma3:4b")
    llm_temperature: float = Field(default=0.1)  # low temp: factual crop Q&A, not creative writing

    # --- Your existing context retrieval API ---
    context_api_base_url: str = Field(default="https://socket.farminsight.dev")
    context_api_retrieve_path: str = Field(default="/rag/retrieve")
    context_api_timeout_seconds: float = Field(default=20.0)

    # --- Retrieval / confidence gating ---
    # Below this confidence, we refuse rather than answer from weak context.
    min_retrieval_confidence: str = Field(default="medium")  # "low" | "medium" | "high"

    # --- Chat history ---
    max_history_turns: int = Field(default=6)  # how many past turns to feed into rewriting/generation

    # --- FastAPI ---
    app_env: str = Field(default="dev")
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. Use this everywhere instead of
    instantiating Settings() directly, so env is parsed once.
    """
    return Settings()
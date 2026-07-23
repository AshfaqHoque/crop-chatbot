from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import configure_logging

def create_app() -> FastAPI:

    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="Crop RAG Chatbot",
        version="0.1.0",
        description="Multilingual RAG chatbot for crop information.",
    )

    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {
            "status": "ok",
            "env": settings.app_env,
            "model": settings.ollama_model,
        }

    return app


app = create_app()
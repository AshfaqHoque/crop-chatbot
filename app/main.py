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

    # NOTE: chat router will be included here in a later step, once
    # app/api/routes/chat.py and the pipeline behind it exist:
    #
    # from app.api.routes.chat import router as chat_router
    # app.include_router(chat_router, prefix="/chat", tags=["chat"])

    return app


app = create_app()
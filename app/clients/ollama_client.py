"""
Factory for LangChain's ChatOllama, so every pipeline stage that needs
an LLM gets it configured the same way instead of instantiating
ChatOllama with slightly different args in five different files.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from app.core.config import get_settings


def get_chat_model(
    *,
    temperature: float | None = None,
    tools: list[BaseTool] | None = None,
) -> BaseChatModel:
    """
    Returns a configured ChatOllama instance.

    - temperature: override the default (e.g. 0.0 for classification
      stages that must be deterministic, vs the configured default for
      free-form answer generation).
    - tools: LangChain tools to bind for function-calling (used by the
      answer-generation stage for unit conversion / calculator).
    """
    settings = get_settings()
    llm: BaseChatModel = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
    )

    if tools:
        llm = llm.bind_tools(tools)

    return llm
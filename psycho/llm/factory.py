"""LLM provider factory — creates the right provider from settings."""

from __future__ import annotations

from loguru import logger

from psycho.config import get_settings

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .ollama_provider import OllamaProvider


def create_provider() -> LLMProvider:
    """
    Read LLM_PROVIDER from settings and return the appropriate provider.

    Supported values:
        "anthropic" → AnthropicProvider (requires ANTHROPIC_API_KEY)
        "ollama"    → OllamaProvider (requires Ollama server running locally)
    """
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        provider = AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
        logger.info(f"Using Anthropic provider: {settings.anthropic_model}")
        return provider

    if settings.llm_provider == "ollama":
        provider = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
        logger.info(f"Using Ollama provider: {settings.ollama_model}")
        return provider

    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
        "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=ollama in .env"
    )

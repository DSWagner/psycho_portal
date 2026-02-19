"""Ollama local model provider — zero API key, runs on your machine."""

from __future__ import annotations

from typing import AsyncIterator

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMProvider, LLMResponse, Message


class OllamaProvider(LLMProvider):
    """
    Ollama local model provider via the OpenAI-compatible REST API.

    No API key required. Runs any model you've pulled with `ollama pull <model>`.
    Default: llama3.2

    Start Ollama: `ollama serve` (usually auto-starts on install)
    Pull a model: `ollama pull llama3.2`
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(120.0),  # local models can be slow
        )
        logger.info(f"OllamaProvider initialized: model={model} @ {base_url}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        )

        payload = {
            "model": self._model,
            "messages": all_messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")
        result = LLMResponse(
            content=content,
            model=self._model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            stop_reason="end_turn",
        )
        logger.debug(
            f"Ollama complete: {result.input_tokens}→{result.output_tokens} tokens"
        )
        return result

    async def stream(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        import json

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        )

        payload = {
            "model": self._model,
            "messages": all_messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using Ollama's embedding endpoint."""
        payload = {"model": self._model, "prompt": text}
        response = await self._client.post("/api/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "ollama"

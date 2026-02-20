"""Anthropic (Claude) LLM provider — uses the official anthropic SDK."""

from __future__ import annotations

from typing import AsyncIterator

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMProvider, LLMResponse, Message


class AnthropicProvider(LLMProvider):
    """
    Claude API provider via the official `anthropic` Python SDK.

    Default model for dev: claude-haiku-4-5-20251001 (cheapest, fast).
    Switch to claude-sonnet-4-6 for production quality.
    """

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        logger.info(f"AnthropicProvider initialized: model={model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=anthropic_messages,
        )

        content = response.content[0].text if response.content else ""
        result = LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "end_turn",
        )
        logger.debug(
            f"Anthropic complete: {result.input_tokens}→{result.output_tokens} tokens"
        )
        return result

    async def stream(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=anthropic_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def complete_with_image(
        self,
        image_data: bytes,
        media_type: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """Extract knowledge from an image using Claude vision."""
        import base64
        b64 = base64.standard_b64encode(image_data).decode("utf-8")
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system or "You are a precise knowledge extractor. Be exhaustive and factual.",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text if response.content else ""

    async def embed(self, text: str) -> list[float]:
        # Anthropic doesn't expose a standalone embedding endpoint in the SDK yet.
        # Phase 2 will add sentence-transformers as the local embedding backend.
        raise NotImplementedError(
            "Embeddings not yet wired for Anthropic provider. "
            "Will be handled by sentence-transformers in Phase 2."
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "anthropic"

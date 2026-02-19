"""Abstract LLM provider interface — swap providers by changing one config line."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Implementors: AnthropicProvider, OllamaProvider
    Both expose the same interface so the agent never knows which is running.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a completion from a list of messages."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream tokens from a completion."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate a dense embedding vector for the given text.

        Used by the semantic memory (ChromaDB) layer.
        Raises NotImplementedError in providers that don't support it natively;
        the MemoryManager falls back to sentence-transformers in that case.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider label ('anthropic', 'ollama', etc.)."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name}>"

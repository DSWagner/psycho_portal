"""Pydantic Settings — reads from .env, provides typed config singleton."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM Provider ──────────────────────────────────────────────
    llm_provider: str = Field(default="anthropic", description="'anthropic' or 'ollama'")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model ID",
    )
    ollama_model: str = Field(default="llama3.2", description="Ollama model name")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )

    # ── Storage Paths ─────────────────────────────────────────────
    data_dir: Path = Field(default=Path("data"))
    db_path: Path = Field(default=Path("data/psycho.db"))
    graph_path: Path = Field(default=Path("data/graph"))
    vector_path: Path = Field(default=Path("data/vectors"))
    journal_path: Path = Field(default=Path("data/journals"))

    # ── Agent Behavior ────────────────────────────────────────────
    max_short_term_messages: int = Field(default=20)
    max_context_memories: int = Field(default=5)
    extraction_enabled: bool = Field(default=True)
    reflection_enabled: bool = Field(default=True)

    # ── TTS (Text-to-Speech) ──────────────────────────────────────
    tts_provider: str = Field(
        default="browser",
        description="'openai', 'elevenlabs', or 'browser' (browser = Web Speech API)",
    )
    openai_api_key: str = Field(default="", description="OpenAI API key (used for TTS)")
    tts_voice: str = Field(
        default="alloy",
        description="TTS voice name. OpenAI: alloy/echo/fable/onyx/nova/shimmer. ElevenLabs: voice ID.",
    )
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID (default: Rachel)"
    )

    # ── API Server ────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"anthropic", "ollama"}
        if v not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return v

    def ensure_data_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for path in [
            self.data_dir,
            self.db_path.parent,
            self.graph_path,
            self.vector_path,
            self.journal_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton. Reads .env on first call."""
    return Settings()

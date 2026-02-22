"""Pydantic Settings — reads from .env, provides typed config singleton."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

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

    # ── Personality ───────────────────────────────────────────────
    # Stored in data/personality.json — these are just initial defaults
    personality_humor: float = Field(default=0.75, description="Humor level 0.0-1.0")
    personality_wit: float = Field(default=0.82, description="Wit level 0.0-1.0")
    personality_directness: float = Field(default=0.88, description="Directness 0.0-1.0")
    personality_warmth: float = Field(default=0.72, description="Warmth 0.0-1.0")
    personality_sass: float = Field(default=0.38, description="Sass 0.0-1.0")
    personality_formality: float = Field(default=0.12, description="Formality 0.0-1.0")
    personality_proactive: float = Field(default=0.82, description="Proactivity 0.0-1.0")
    personality_empathy: float = Field(default=0.78, description="Empathy 0.0-1.0")

    # ── Proactive System ──────────────────────────────────────────
    proactive_enabled: bool = Field(default=True, description="Enable proactive check-ins and reminders")
    proactive_scheduler_interval: int = Field(default=60, description="Scheduler tick interval in seconds")
    checkin_enabled: bool = Field(default=True, description="Enable morning/evening check-ins")

    # ── Google Calendar (optional) ────────────────────────────────
    google_calendar_credentials: str = Field(
        default="",
        description="Path to Google Calendar OAuth2 credentials.json (optional)",
    )

    # ── TTS (Text-to-Speech) ──────────────────────────────────────
    tts_provider: str = Field(
        default="browser",
        description="'openai', 'elevenlabs', 'local', or 'browser'",
    )
    openai_api_key: str = Field(default="", description="OpenAI API key (used for TTS)")
    tts_voice: str = Field(
        default="alloy",
        description="TTS voice name. OpenAI: alloy/echo/fable/onyx/nova/shimmer",
    )
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID (default: Rachel)"
    )

    # ── Local TTS ─────────────────────────────────────────────────
    local_tts_backend: str = Field(
        default="pyttsx3",
        description="Local TTS backend: 'kokoro' | 'coqui' | 'pyttsx3'",
    )
    local_tts_voice: str = Field(
        default="",
        description="Local TTS voice (kokoro: af_heart/af_sky/am_adam, pyttsx3: voice name)",
    )
    local_tts_speed: float = Field(default=1.0, description="TTS speech speed multiplier")

    # ── STT (Speech-to-Text) ──────────────────────────────────────
    stt_provider: str = Field(
        default="browser",
        description="'browser' (Web Speech API) or 'whisper_local'",
    )

    # ── Local Whisper STT ─────────────────────────────────────────
    whisper_model: str = Field(
        default="base",
        description="Whisper model size: tiny | base | small | medium | large-v3",
    )
    whisper_device: str = Field(
        default="cpu",
        description="Whisper inference device: 'cpu' | 'cuda' | 'auto'",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="Whisper compute type: 'int8' | 'float16' | 'float32'",
    )
    whisper_language: str = Field(
        default="",
        description="Whisper language (empty = auto-detect)",
    )
    whisper_backend: str = Field(
        default="faster_whisper",
        description="Whisper backend: 'faster_whisper' | 'openai_whisper'",
    )

    # ── Web Search ────────────────────────────────────────────────
    web_search_enabled: bool = Field(
        default=True,
        description="Inject live web results for queries needing current data",
    )
    brave_api_key: str = Field(
        default="",
        description="Brave Search API key (optional, falls back to DuckDuckGo)",
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

    @field_validator("tts_provider")
    @classmethod
    def validate_tts(cls, v: str) -> str:
        allowed = {"browser", "openai", "elevenlabs", "local"}
        if v not in allowed:
            raise ValueError(f"tts_provider must be one of {allowed}, got '{v}'")
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

    def get_personality_path(self) -> Path:
        return self.data_dir / "personality.json"

    def get_initial_personality_dict(self) -> dict:
        """Return initial personality values from .env settings."""
        return {
            "humor_level": self.personality_humor,
            "wit_level": self.personality_wit,
            "directness_level": self.personality_directness,
            "warmth_level": self.personality_warmth,
            "sass_level": self.personality_sass,
            "formality_level": self.personality_formality,
            "proactive_level": self.personality_proactive,
            "empathy_level": self.personality_empathy,
            "curiosity_level": 0.68,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton. Reads .env on first call."""
    return Settings()

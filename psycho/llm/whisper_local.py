"""
LocalWhisperSTT — local speech-to-text using faster-whisper or openai-whisper.

This is the backend counterpart to the browser's Web Speech API.
Audio (WebM/WAV/MP3) is sent from the browser to the server via a POST endpoint,
transcribed locally, and returned as text.

Supports two backends:
  - faster-whisper (recommended): faster, lower memory, ONNX-accelerated
  - openai-whisper: the original, higher accuracy at same model size

Configure in .env:
    STT_PROVIDER=whisper_local
    WHISPER_MODEL=base          # tiny | base | small | medium | large-v3
    WHISPER_DEVICE=cpu          # cpu | cuda | auto
    WHISPER_COMPUTE_TYPE=int8   # int8 | float16 | float32
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger


class LocalWhisperSTT:
    """
    Local Whisper speech-to-text transcription.

    Lazy-loads the model on first use to avoid startup delay.
    Thread-safe via asyncio.Lock for concurrent requests.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,  # None = auto-detect
        backend: str = "faster_whisper",  # "faster_whisper" | "openai_whisper"
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._backend = backend
        self._model = None
        self._lock = None  # Initialized on first async call

    async def _get_lock(self):
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def ensure_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        lock = await self._get_lock()
        async with lock:
            if self._model is not None:  # Double-check after lock
                return

            if self._backend == "faster_whisper":
                await self._load_faster_whisper()
            else:
                await self._load_openai_whisper()

    async def _load_faster_whisper(self) -> None:
        try:
            from faster_whisper import WhisperModel
            logger.info(
                f"Loading faster-whisper model '{self._model_size}' "
                f"on {self._device} ({self._compute_type})"
            )
            # Run in thread to avoid blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                ),
            )
            logger.info("faster-whisper model loaded")
        except ImportError:
            logger.warning("faster-whisper not installed. Install: pip install faster-whisper")
            raise

    async def _load_openai_whisper(self) -> None:
        try:
            import whisper
            logger.info(f"Loading openai-whisper model '{self._model_size}'")
            import asyncio
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: whisper.load_model(self._model_size),
            )
            logger.info("openai-whisper model loaded")
        except ImportError:
            logger.warning("openai-whisper not installed. Install: pip install openai-whisper")
            raise

    async def transcribe(self, audio_data: bytes, audio_format: str = "webm") -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_data: Raw audio bytes (WebM, WAV, MP3, etc.)
            audio_format: File format hint for ffmpeg

        Returns:
            Transcribed text string.
        """
        await self.ensure_loaded()

        import asyncio

        # Write audio to a temp file (whisper needs a file path)
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}", delete=False
        ) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_event_loop()
            if self._backend == "faster_whisper":
                text = await loop.run_in_executor(
                    None, self._transcribe_faster, tmp_path
                )
            else:
                text = await loop.run_in_executor(
                    None, self._transcribe_openai, tmp_path
                )
            return text.strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _transcribe_faster(self, audio_path: str) -> str:
        """Synchronous faster-whisper transcription."""
        segments, info = self._model.transcribe(
            audio_path,
            language=self._language,
            beam_size=5,
            vad_filter=True,  # Voice activity detection
        )
        text_parts = [segment.text for segment in segments]
        logger.debug(
            f"Whisper transcribed {info.duration:.1f}s audio "
            f"(lang={info.language}, prob={info.language_probability:.2f})"
        )
        return " ".join(text_parts)

    def _transcribe_openai(self, audio_path: str) -> str:
        """Synchronous openai-whisper transcription."""
        result = self._model.transcribe(
            audio_path,
            language=self._language,
            fp16=False,  # CPU-safe
        )
        return result["text"]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_info(self) -> str:
        return f"{self._backend}/{self._model_size} on {self._device}"

    @classmethod
    def from_settings(cls, settings) -> "LocalWhisperSTT":
        """Create from Settings object."""
        return cls(
            model_size=getattr(settings, "whisper_model", "base"),
            device=getattr(settings, "whisper_device", "cpu"),
            compute_type=getattr(settings, "whisper_compute_type", "int8"),
            language=getattr(settings, "whisper_language", None) or None,
            backend=getattr(settings, "whisper_backend", "faster_whisper"),
        )

"""
LocalTTSProvider — local text-to-speech synthesis.

Provides fully-offline TTS as an alternative to browser, OpenAI, or ElevenLabs TTS.

Backends (in order of quality):
1. Kokoro (kokoro-onnx) — high quality, 82M params, fully local, ~300MB
2. Coqui TTS — good quality, heavier
3. pyttsx3 — system TTS, zero download, low quality but always works

Configure in .env:
    TTS_PROVIDER=local
    LOCAL_TTS_BACKEND=kokoro       # kokoro | coqui | pyttsx3
    LOCAL_TTS_VOICE=af_heart       # kokoro: af_heart, af_sky, am_adam, etc.
    LOCAL_TTS_SPEED=1.0            # Speech speed multiplier
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Optional

from loguru import logger


class LocalTTSProvider:
    """
    Local TTS synthesis — returns audio bytes (WAV/MP3) from text.

    Drop-in replacement for the cloud TTS providers.
    Lazy-loads the model on first synthesis call.
    """

    def __init__(
        self,
        backend: str = "pyttsx3",  # "kokoro" | "coqui" | "pyttsx3"
        voice: str = "",
        speed: float = 1.0,
        sample_rate: int = 24000,
    ) -> None:
        self._backend = backend
        self._voice = voice or self._default_voice(backend)
        self._speed = speed
        self._sample_rate = sample_rate
        self._engine = None  # Lazy-loaded
        self._lock = None

    def _default_voice(self, backend: str) -> str:
        defaults = {
            "kokoro": "af_heart",      # American female, warm
            "coqui": "tts_models/en/ljspeech/tacotron2-DDC",
            "pyttsx3": "",
        }
        return defaults.get(backend, "")

    async def _get_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech. Returns WAV audio bytes.
        """
        if not text.strip():
            return b""

        lock = await self._get_lock()
        async with lock:
            loop = asyncio.get_event_loop()
            if self._backend == "kokoro":
                return await loop.run_in_executor(None, self._synth_kokoro, text)
            elif self._backend == "coqui":
                return await loop.run_in_executor(None, self._synth_coqui, text)
            else:
                return await loop.run_in_executor(None, self._synth_pyttsx3, text)

    # ── Kokoro backend ─────────────────────────────────────────────

    def _synth_kokoro(self, text: str) -> bytes:
        try:
            if self._engine is None:
                from kokoro_onnx import Kokoro
                logger.info(f"Loading Kokoro TTS model (voice: {self._voice})")
                self._engine = Kokoro("kokoro-v0_19.onnx", "voices.bin")
                logger.info("Kokoro TTS loaded")

            import soundfile as sf
            samples, sample_rate = self._engine.create(
                text,
                voice=self._voice,
                speed=self._speed,
                lang="en-us",
            )
            buf = io.BytesIO()
            sf.write(buf, samples, sample_rate, format="WAV")
            return buf.getvalue()

        except ImportError:
            logger.warning("kokoro-onnx not installed. Falling back to pyttsx3.")
            return self._synth_pyttsx3(text)
        except Exception as e:
            logger.error(f"Kokoro TTS failed: {e}")
            return self._synth_pyttsx3(text)

    # ── Coqui TTS backend ──────────────────────────────────────────

    def _synth_coqui(self, text: str) -> bytes:
        try:
            if self._engine is None:
                from TTS.api import TTS
                logger.info(f"Loading Coqui TTS model: {self._voice}")
                self._engine = TTS(model_name=self._voice)
                logger.info("Coqui TTS loaded")

            with io.BytesIO() as buf:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                self._engine.tts_to_file(text=text, file_path=tmp_path)
                with open(tmp_path, "rb") as f:
                    data = f.read()
                os.unlink(tmp_path)
                return data

        except ImportError:
            logger.warning("TTS (Coqui) not installed. Falling back to pyttsx3.")
            return self._synth_pyttsx3(text)
        except Exception as e:
            logger.error(f"Coqui TTS failed: {e}")
            return self._synth_pyttsx3(text)

    # ── pyttsx3 backend (always available) ─────────────────────────

    def _synth_pyttsx3(self, text: str) -> bytes:
        """
        System TTS via pyttsx3. Zero model download, works on any OS.
        Returns WAV bytes by saving to a temp file.
        """
        try:
            import pyttsx3
            import tempfile
            import os

            if self._engine is None or not isinstance(self._engine, type(pyttsx3.init())):
                engine = pyttsx3.init()
                engine.setProperty("rate", int(engine.getProperty("rate") * self._speed))
                if self._voice:
                    voices = engine.getProperty("voices")
                    for v in voices:
                        if self._voice.lower() in v.name.lower():
                            engine.setProperty("voice", v.id)
                            break
                self._engine = engine

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()

            with open(tmp_path, "rb") as f:
                data = f.read()
            os.unlink(tmp_path)
            return data

        except ImportError:
            logger.warning("pyttsx3 not installed. Install: pip install pyttsx3")
            return b""
        except Exception as e:
            logger.error(f"pyttsx3 TTS failed: {e}")
            return b""

    @property
    def provider_name(self) -> str:
        return f"local_{self._backend}"

    @property
    def is_available(self) -> bool:
        """Quick check if the backend is likely available."""
        try:
            if self._backend == "kokoro":
                import importlib
                return importlib.util.find_spec("kokoro_onnx") is not None
            elif self._backend == "coqui":
                import importlib
                return importlib.util.find_spec("TTS") is not None
            else:
                import importlib
                return importlib.util.find_spec("pyttsx3") is not None
        except Exception:
            return False

    @classmethod
    def from_settings(cls, settings) -> "LocalTTSProvider":
        """Create from Settings object."""
        return cls(
            backend=getattr(settings, "local_tts_backend", "pyttsx3"),
            voice=getattr(settings, "local_tts_voice", ""),
            speed=getattr(settings, "local_tts_speed", 1.0),
        )

"""
LocalTTSProvider — local text-to-speech synthesis.

Provides fully-offline TTS as an alternative to browser, OpenAI, or ElevenLabs.

Backends (in order of quality):
1. kokoro  (hexgrad/Kokoro-82M, v1.0)  — state-of-the-art open-source TTS,
           82M params, PyTorch, auto-downloads from HuggingFace (~450MB first run)
           Voices: af_heart, af_bella, af_nicole, am_adam, bm_george, bm_lewis, ...
2. kokoro_onnx  (legacy ONNX v0.19)    — lighter but lower quality
3. pyttsx3      (system TTS)           — zero download, robotic, last resort

Configure in .env:
    TTS_PROVIDER=local
    LOCAL_TTS_BACKEND=kokoro          # kokoro | kokoro_onnx | pyttsx3
    LOCAL_TTS_VOICE=bm_george         # see VOICES below
    LOCAL_TTS_SPEED=1.0               # 0.5–2.0 speech rate multiplier

Recommended voices (Kokoro v1):
    FEMALE (American)  : af_heart, af_bella, af_nicole, af_sky, af_sarah
    MALE   (American)  : am_adam, am_michael, am_echo, am_liam
    FEMALE (British)   : bf_emma, bf_alice, bf_isabella, bf_lily
    MALE   (British)   : bm_george, bm_lewis, bm_daniel, bm_fable
    (bm_george sounds remarkably like JARVIS — highly recommended)
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Optional

from loguru import logger

# Suppress the HuggingFace symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class LocalTTSProvider:
    """
    Local TTS synthesis — returns audio bytes (WAV) from text.

    Drop-in replacement for the cloud TTS providers.
    Lazy-loads the model on first synthesis call (auto-downloads on first use).
    Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        backend: str = "kokoro",
        voice: str = "",
        speed: float = 1.0,
        sample_rate: int = 24000,
    ) -> None:
        self._backend = backend
        self._voice = voice or self._default_voice(backend)
        self._speed = speed
        self._sample_rate = sample_rate
        self._engine = None   # lazy-loaded
        self._lock: Optional[asyncio.Lock] = None

    def _default_voice(self, backend: str) -> str:
        return {
            "kokoro":      "bm_george",   # British male — very JARVIS-like
            "kokoro_onnx": "af_heart",    # American female — warm
            "pyttsx3":     "",
        }.get(backend, "")

    async def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech. Returns WAV audio bytes."""
        if not text.strip():
            return b""

        lock = await self._get_lock()
        async with lock:
            loop = asyncio.get_event_loop()
            if self._backend == "kokoro":
                return await loop.run_in_executor(None, self._synth_kokoro_v1, text)
            elif self._backend == "kokoro_onnx":
                return await loop.run_in_executor(None, self._synth_kokoro_onnx, text)
            else:
                return await loop.run_in_executor(None, self._synth_pyttsx3, text)

    # ── Kokoro v1 (hexgrad/Kokoro-82M, PyTorch) ────────────────────────────────

    def _synth_kokoro_v1(self, text: str) -> bytes:
        """
        Best quality — uses the hexgrad Kokoro-82M v1.0 model.
        Auto-downloads from HuggingFace on first run (~450MB, cached permanently).
        """
        try:
            if self._engine is None:
                from kokoro import KPipeline
                logger.info(
                    f"Loading Kokoro v1 TTS (voice={self._voice}). "
                    "First run downloads ~450MB from HuggingFace…"
                )
                self._engine = KPipeline(
                    lang_code="a",            # 'a' = American English (works for British too)
                    repo_id="hexgrad/Kokoro-82M",
                )
                logger.info("Kokoro v1 TTS model loaded and ready.")

            import numpy as np
            import soundfile as sf

            chunks = []
            for _gs, _ps, audio in self._engine(
                text,
                voice=self._voice,
                speed=self._speed,
                split_pattern=r"[.!?;:\n]+",
            ):
                chunks.append(audio)

            if not chunks:
                return b""

            full_audio = np.concatenate(chunks)
            buf = io.BytesIO()
            sf.write(buf, full_audio, 24000, format="WAV")
            return buf.getvalue()

        except ImportError:
            logger.warning(
                "kokoro not installed. Run: pip install kokoro soundfile\n"
                "Falling back to kokoro_onnx or pyttsx3."
            )
            return self._synth_kokoro_onnx(text)
        except Exception as e:
            logger.error(f"Kokoro v1 TTS error: {e}")
            return self._synth_kokoro_onnx(text)

    # ── Kokoro ONNX (legacy v0.19) ─────────────────────────────────────────────

    def _synth_kokoro_onnx(self, text: str) -> bytes:
        """
        Legacy ONNX backend — lighter than v1 but lower quality.
        Requires: pip install kokoro-onnx soundfile
        Model files must be downloaded manually or via the kokoro-onnx docs.
        """
        try:
            if self._engine is None:
                from kokoro_onnx import Kokoro
                # Try to auto-download via huggingface_hub if available
                model_path, voices_path = self._resolve_kokoro_onnx_files()
                logger.info(f"Loading Kokoro ONNX TTS (voice={self._voice})")
                self._engine = Kokoro(model_path, voices_path)
                logger.info("Kokoro ONNX TTS loaded.")

            import soundfile as sf
            samples, sample_rate = self._engine.create(
                text,
                voice=self._voice or "af_heart",
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
            logger.error(f"Kokoro ONNX TTS error: {e}")
            return self._synth_pyttsx3(text)

    def _resolve_kokoro_onnx_files(self) -> tuple[str, str]:
        """Return (model_path, voices_path), downloading from HuggingFace if needed."""
        model_file = Path("kokoro-v0_19.onnx")
        voices_file = Path("voices.bin")

        if model_file.exists() and voices_file.exists():
            return str(model_file), str(voices_file)

        try:
            from huggingface_hub import hf_hub_download
            logger.info("Downloading Kokoro ONNX model files from HuggingFace…")
            mp = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v0_19.onnx")
            vp = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices.bin")
            return mp, vp
        except Exception as e:
            logger.warning(f"Could not auto-download kokoro-onnx files: {e}")
            return str(model_file), str(voices_file)  # will fail gracefully in caller

    # ── pyttsx3 (system TTS — always available, low quality) ──────────────────

    def _synth_pyttsx3(self, text: str) -> bytes:
        """System TTS via pyttsx3. Zero model download, works everywhere."""
        try:
            import pyttsx3
            import tempfile

            engine = pyttsx3.init()
            engine.setProperty("rate", int(engine.getProperty("rate") * self._speed))
            if self._voice:
                for v in engine.getProperty("voices"):
                    if self._voice.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            data = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)
            return data

        except ImportError:
            logger.warning("pyttsx3 not installed. Install: pip install pyttsx3")
            return b""
        except Exception as e:
            logger.error(f"pyttsx3 TTS error: {e}")
            return b""

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return f"local_{self._backend}"

    @property
    def is_available(self) -> bool:
        """Quick check — does the backend package appear to be installed?"""
        import importlib.util
        pkg = {
            "kokoro":      "kokoro",
            "kokoro_onnx": "kokoro_onnx",
            "pyttsx3":     "pyttsx3",
        }.get(self._backend, self._backend)
        return importlib.util.find_spec(pkg) is not None

    @classmethod
    def from_settings(cls, settings) -> "LocalTTSProvider":
        return cls(
            backend=getattr(settings, "local_tts_backend", "kokoro"),
            voice=getattr(settings, "local_tts_voice", ""),
            speed=getattr(settings, "local_tts_speed", 1.0),
        )

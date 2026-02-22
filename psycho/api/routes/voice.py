"""Voice routes — GET /api/voice/config, POST /api/voice/tts, POST /api/voice/stt."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, Response

router = APIRouter(prefix="/api/voice", tags=["voice"])

# Lazy-loaded singletons — created once, reused for all requests
_whisper_stt = None
_local_tts = None


def _get_whisper():
    global _whisper_stt
    if _whisper_stt is None:
        from psycho.config import get_settings
        from psycho.llm.whisper_local import LocalWhisperSTT
        s = get_settings()
        _whisper_stt = LocalWhisperSTT.from_settings(s)
    return _whisper_stt


def _get_local_tts():
    global _local_tts
    if _local_tts is None:
        from psycho.config import get_settings
        from psycho.llm.local_tts import LocalTTSProvider
        s = get_settings()
        _local_tts = LocalTTSProvider.from_settings(s)
    return _local_tts


@router.get("/config")
async def voice_config():
    """
    Return the active TTS provider so the frontend knows whether to
    expect audio from the server or fall back to browser SpeechSynthesis.
    """
    from psycho.config import get_settings
    s = get_settings()

    if s.tts_provider == "openai" and s.openai_api_key:
        provider = "openai"
        voice = s.tts_voice or "alloy"
    elif s.tts_provider == "elevenlabs" and s.elevenlabs_api_key:
        provider = "elevenlabs"
        voice = s.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"
    elif s.tts_provider == "local":
        provider = "local"
        voice = s.local_tts_voice or s.local_tts_backend
    else:
        provider = "browser"
        voice = "browser"

    return {
        "provider": provider,
        "voice": voice,
        "stt_provider": s.stt_provider,
    }


@router.post("/tts")
async def text_to_speech(request_body: dict):
    """
    Convert text to speech audio and return MP3 bytes.

    Tries providers in order:
      1. OpenAI TTS  (if tts_provider=openai + openai_api_key set)
      2. ElevenLabs  (if tts_provider=elevenlabs + elevenlabs_api_key set)
      3. 204 No Content → client falls back to browser SpeechSynthesis
    """
    from psycho.config import get_settings
    import httpx

    s = get_settings()
    text = (request_body.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})

    # ── OpenAI TTS ────────────────────────────────────────────────
    if s.tts_provider == "openai" and s.openai_api_key:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {s.openai_api_key}"},
                    json={
                        "model": "tts-1",
                        "input": text[:4096],
                        "voice": s.tts_voice or "alloy",
                        "response_format": "mp3",
                    },
                )
            if r.status_code == 200:
                return Response(content=r.content, media_type="audio/mpeg")
            return JSONResponse(status_code=502, content={"error": f"OpenAI TTS error {r.status_code}"})
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    # ── ElevenLabs TTS ────────────────────────────────────────────
    if s.tts_provider == "elevenlabs" and s.elevenlabs_api_key:
        voice_id = s.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": s.elevenlabs_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text[:5000],
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    },
                )
            if r.status_code == 200:
                return Response(content=r.content, media_type="audio/mpeg")
            return JSONResponse(status_code=502, content={"error": f"ElevenLabs error {r.status_code}"})
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    # ── Local TTS ─────────────────────────────────────────────────
    if s.tts_provider == "local":
        try:
            tts = _get_local_tts()
            audio = await tts.synthesize(text)
            if audio:
                return Response(content=audio, media_type="audio/wav")
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": str(e)})

    # ── No server TTS configured ─────────────────────────────────
    # Return 204 — client falls back to browser SpeechSynthesis
    return Response(status_code=204)


@router.post("/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    format: str = "webm",
):
    """
    Transcribe audio to text using local Whisper.
    Only active when STT_PROVIDER=whisper_local in .env.

    Send audio as multipart/form-data with field name 'audio'.
    Returns { "text": "transcribed text" }
    """
    from psycho.config import get_settings
    s = get_settings()

    if s.stt_provider != "whisper_local":
        return JSONResponse(
            status_code=404,
            content={"error": "Local Whisper STT not configured. Set STT_PROVIDER=whisper_local"},
        )

    try:
        audio_data = await audio.read()
        if not audio_data:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})

        whisper = _get_whisper()
        text = await whisper.transcribe(audio_data, audio_format=format)
        return {"text": text, "model": whisper.model_info}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

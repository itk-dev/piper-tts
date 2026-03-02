import json
import logging
import sys
from langdetect import detect, detect_langs

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Security,
    Body,
)
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import io
import wave
from piper import PiperVoice, SynthesisConfig
import os
from prometheus_fastapi_instrumentator import Instrumentator

# Import pydub for MP3 conversion
from pydub import AudioSegment

# configure CORS
from fastapi.middleware.cors import CORSMiddleware

# Configure logging to output to stdout/stderr
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # This sends logs to stdout
)

# Create a logger for your application
logger = logging.getLogger(__name__)

# Start the API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Use GPU.
CUDA_ENABLED = os.environ.get("CUDA_ENABLED", "false").lower() == "true"

# Define API key.
API_KEY = os.environ.get("API_KEY", "CHANGE_ME")

# Create the security scheme for Bearer tokens
bearer_scheme = HTTPBearer(auto_error=False)


def get_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    """
    Validate the Bearer token from the Authorization header.

    Args:
        credentials: The Bearer token credentials extracted from the Authorization header

    Returns:
        The validated token if authentication is successful

    Raises:
        HTTPException: If authentication fails
    """
    if credentials is None:
        raise HTTPException(
            status_code=401, detail="Missing Authorization header with Bearer token"
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Authorization header must use Bearer scheme"
        )

    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

    return credentials.credentials


# Initialize Prometheus monitoring
Instrumentator().instrument(app).expose(
    app,
    include_in_schema=False,
    dependencies=[Depends(get_bearer_token, use_cache=False)],
)


# Language to voice mapping
# Load voice mapping from the environment
voices_str = os.environ.get("VOICES")
if voices_str:
    voices_config = json.loads(voices_str)
    # Convert array to lookup dictionaries
    LANGUAGE_VOICE_MAPPING = {
        voice["language"]: voice["path"] for voice in voices_config
    }
    VOICE_ID_MAPPING = {voice["id"]: voice["path"] for voice in voices_config}
    # Remove path from voices before exposing to API
    AVAILABLE_VOICES = [
        {k: v for k, v in voice.items() if k != "path"} for voice in voices_config
    ]
else:
    raise Exception("VOICES environment variable is not set")

# Default language if detection fails
DEFAULT_LANGUAGE = "da"

# Pre-load all voice models at startup to avoid reloading on every request
LOADED_VOICE_MODELS = {}
for _voice_cfg in voices_config:
    _path = _voice_cfg["path"]
    if _path not in LOADED_VOICE_MODELS:
        logger.info(f"Loading voice model: {_path}")
        LOADED_VOICE_MODELS[_path] = PiperVoice.load(_path, use_cuda=CUDA_ENABLED)
logger.info(f"Loaded {len(LOADED_VOICE_MODELS)} voice model(s)")


def detect_language(text):
    """
    Detect language with Danish prioritization over Norwegian
    """
    try:
        # Get all possible languages with confidence scores
        possible_langs = detect_langs(text)

        logger.info(f"Possible languages detected: {possible_langs}")

        # Otherwise return the highest confidence language
        return detect(text)
    except:
        logger.warning(
            f"Language detection failed: {e}, using default language: {DEFAULT_LANGUAGE}"
        )
        return DEFAULT_LANGUAGE


@app.get("/", include_in_schema=False)
def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    status = {"api_status": "ok"}

    return JSONResponse(content=status, status_code=200)


@app.get("/audio/voices")
async def list_voices(token: str = Depends(get_bearer_token, use_cache=False)):
    """
    List available voices for text-to-speech generation.
    Compatible with OpenAI's TTS API.
    """
    return {"voices": AVAILABLE_VOICES}


@app.get("/audio/models")
async def list_models(token: str = Depends(get_bearer_token, use_cache=False)):
    return {"models": [{"id": "piper-tts-1"}]}


# OpenAI TTS endpoint that matches their API
@app.post("/audio/speech")
async def create_speech(
    model: str = Body(..., description="This has no effect"),
    voice: str = Body(..., description="Select voice to use"),
    input: str = Body(..., description="The text to generate speech for"),
    response_format: str = Body(
        "mp3", description="The format of the audio response (mp3, wav, or pcm)"
    ),
    speed: float = Body(1.0, description="The speed of the generated audio"),
    auto_detect_language: bool = Body(
        True, description="Automatically detect language and use appropriate voice"
    ),
    token: str = Depends(get_bearer_token, use_cache=False),
):
    """
    Creates speech from the input text using the specified voice.
    Compatible with OpenAI's TTS API.
    """
    # If auto-detect is enabled, try to detect the language and use the appropriate voice
    if auto_detect_language:
        try:
            detected_lang = detect_language(input)
            # First, try the detected language
            voice_path = LANGUAGE_VOICE_MAPPING.get(detected_lang)

            # If not found, try the voice parameter as ID or language
            if not voice_path:
                voice_path = VOICE_ID_MAPPING.get(
                    voice.lower()
                ) or LANGUAGE_VOICE_MAPPING.get(voice.lower())

            # If still not found, use default
            if not voice_path:
                voice_path = LANGUAGE_VOICE_MAPPING.get(DEFAULT_LANGUAGE)

            logger.info(
                f"Auto-detected language: {detected_lang}, using voice: {voice_path}"
            )
        except Exception as e:
            # If language detection fails, use the requested voice
            voice_path = LANGUAGE_VOICE_MAPPING.get(
                voice.lower(), LANGUAGE_VOICE_MAPPING.get(DEFAULT_LANGUAGE)
            )
            logger.error(
                f"Language detection failed: {str(e)}, using requested voice: {voice_path}"
            )
    else:
        voice_path = None

    logger.info(
        f"Processing TTS request with voice: {voice_path}, speed: {speed}, format: {response_format}"
    )

    # Configure synthesis parameters
    syn_config = SynthesisConfig(
        volume=1.0,
        length_scale=1.0 / speed,  # Adjust length scale based on speed
        noise_scale=0.0,
        noise_w_scale=0.0,
        normalize_audio=True,
    )

    try:
        # Look up the pre-loaded voice model
        voice_model = LOADED_VOICE_MODELS.get(voice_path)
        if voice_model is None:
            raise HTTPException(
                status_code=400,
                detail=f"Voice model not found for path: {voice_path}",
            )

        # Create an in-memory file-like object to store the WAV audio
        wav_buffer = io.BytesIO()

        # Generate WAV audio
        with wave.open(wav_buffer, "wb") as wav_file:
            voice_model.synthesize_wav(input, wav_file, syn_config=syn_config)

        # Reset the buffer position to the beginning
        wav_buffer.seek(0)

        fmt = response_format.lower()
        if fmt == "mp3":
            audio = AudioSegment.from_wav(wav_buffer)
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3")
            mp3_buffer.seek(0)

            return StreamingResponse(
                mp3_buffer,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=speech.mp3"},
            )
        elif fmt == "pcm":
            # Read raw PCM samples from the WAV buffer (skip the WAV header)
            with wave.open(wav_buffer, "rb") as wav_reader:
                pcm_data = wav_reader.readframes(wav_reader.getnframes())
            pcm_buffer = io.BytesIO(pcm_data)

            return StreamingResponse(
                pcm_buffer,
                media_type="audio/pcm",
                headers={"Content-Disposition": "attachment; filename=speech.pcm"},
            )
        else:
            return StreamingResponse(
                wav_buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"},
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Speech generation failed: {str(e)}"
        )

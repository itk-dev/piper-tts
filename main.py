import json
from langdetect import detect

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

# Import pydub for MP3 conversion
from pydub import AudioSegment

# Start the API
app = FastAPI()

# configure CORS
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/", include_in_schema=False)
def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    status = {"api_status": "ok"}

    return JSONResponse(content=status, status_code=200)

# Language to voice mapping
# Map language codes to the appropriate voice model
try:
    language_mapping_str = os.environ.get("LANGUAGE_VOICE_MAPPING")
    if language_mapping_str:
        LANGUAGE_VOICE_MAPPING = json.loads(language_mapping_str)
    else:
        LANGUAGE_VOICE_MAPPING = {
            "da": "./voices/da_DK-talesyntese-medium.onnx",
            "en": "./voices/en_US-amy-medium.onnx",
            "gb": "./voices/en_GB-alan-medium.onnx"
        }
except json.JSONDecodeError:
    # If JSON parsing fails, use default mapping
    print("Error parsing LANGUAGE_VOICE_MAPPING from environment, using defaults")
    LANGUAGE_VOICE_MAPPING = {
        "da": "./voices/da_DK-talesyntese-medium.onnx",
        "en": "./voices/en_US-amy-medium.onnx",
        "gb": "./voices/en_GB-alan-medium.onnx"
    }


# Default language if detection fails
DEFAULT_LANGUAGE = "da"

# OpenAI TTS endpoint that matches their API
@app.post("/audio/speech")
async def create_speech(
    model: str = Body(..., description="This has no effect"),
    voice: str = Body(..., description="This has no effect"),
    input: str = Body(..., description="The text to generate speech for"),
    response_format: str = Body(
        "mp3", description="The format of the audio response (mp3 or wav)"
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
            detected_lang = detect(input)
            # Use the detected language voice if available, otherwise fallback to default
            voice_path = LANGUAGE_VOICE_MAPPING.get(
                detected_lang, LANGUAGE_VOICE_MAPPING.get(DEFAULT_LANGUAGE)
            )
            print(f"Detected language: {detected_lang}, using voice: {voice_path}")
        except Exception as e:
            # If language detection fails, use the requested voice
            voice_path = LANGUAGE_VOICE_MAPPING.get(
                voice.lower(), LANGUAGE_VOICE_MAPPING.get(DEFAULT_LANGUAGE)
            )
            print(f"Language detection failed: {str(e)}, using requested voice")
    else:
        # Use the voice specified in the request
        voice_path = LANGUAGE_VOICE_MAPPING.get(
            voice.lower(), LANGUAGE_VOICE_MAPPING.get(DEFAULT_LANGUAGE)
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
        # Load the voice model
        voice_model = PiperVoice.load(voice_path, use_cuda=CUDA_ENABLED)

        # Create an in-memory file-like object to store the WAV audio
        wav_buffer = io.BytesIO()

        # Generate WAV audio
        with wave.open(wav_buffer, "wb") as wav_file:
            voice_model.synthesize_wav(input, wav_file, syn_config=syn_config)

        # Reset the buffer position to the beginning
        wav_buffer.seek(0)

        # If MP3 format is requested, convert WAV to MP3
        if response_format.lower() == "mp3":
            # Convert WAV to MP3 using pydub
            audio = AudioSegment.from_wav(wav_buffer)

            # Create a new buffer for the MP3 data
            mp3_buffer = io.BytesIO()

            # Export as MP3
            audio.export(mp3_buffer, format="mp3")

            # Reset the MP3 buffer position
            mp3_buffer.seek(0)

            # Return the MP3 audio as a streaming response
            return StreamingResponse(
                mp3_buffer,
                media_type="audio/mpeg",
                headers={"Content-Disposition": f"attachment; filename=speech.mp3"},
            )
        else:
            # Return WAV format (reset buffer position first)
            wav_buffer.seek(0)
            return StreamingResponse(
                wav_buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename=speech.wav"},
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Speech generation failed: {str(e)}"
        )

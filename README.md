[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Piper TTS Docker

A containerized API for text-to-speech conversion using Piper TTS with an OpenAI-compatible endpoint.

## Features

- OpenAI TTS API compatible endpoint
- Automatic language detection
- Support for multiple voices
- MP3 and WAV audio formats
- Adjustable speech speed
- Containerized for easy deployment
- API key authentication

## Voices

Voice samples are available at: https://rhasspy.github.io/piper-samples/

### Download Voices

```shell
python3 -m piper.download_voices --download-dir voices da_DK-talesyntese-medium en_GB-alan-medium en_US-amy-medium
``` 

## Configuration

The following environment variables can be configured:

- `API_KEY`: Authentication key for the API (default: "CHANGE_ME")
- `CUDA_ENABLED`: Enable GPU acceleration (default: "false")
- `LANGUAGE_VOICE_MAPPING`: JSON mapping of language codes to voice models (optional)

Default language mapping:

```json 
{
  "da": "./voices/da_DK-talesyntese-medium.onnx",
  "en": "./voices/en_US-amy-medium.onnx",
  "gb": "./voices/en_GB-alan-medium.onnx"
}
``` 

## Usage

### Docker Compose

Use the provided docker-compose files to start the service:

```shell
# Development
docker-compose up
# Server deployment
docker-compose -f docker-compose.server.yml up -d
``` 

### API Endpoints

#### Health Check

```
GET /health
``` 

#### Generate Speech

```
POST /audio/speech
``` 

Request body:

```json 
{
  "model": "tts-1",
  "voice": "en",
  "input": "The text you want to convert to speech",
  "response_format": "mp3",
  "speed": 1.0,
  "auto_detect_language": true
}
``` 

Authentication: Bearer token with your configured API_KEY

__Note__: That the model and voice do not have any effect at the current release. They are there to be compatible with
openAI's api.

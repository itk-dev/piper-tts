
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg for MP3 conversion
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    python3-dev \
    libasound2-dev \
    portaudio19-dev \
    python3-pyaudio \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and lock file
COPY pyproject.toml ./

# Install dependencies
RUN pip install --upgrade pip \
    && pip install .

COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
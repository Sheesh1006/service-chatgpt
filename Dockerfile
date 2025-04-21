FROM python:3.12-slim

# Install system dependencies including ffmpeg and OpenCV support
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       build-essential \
       ffmpeg \
       python3-opencv \
       libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./

# Default command
CMD ["python", "main.py"]

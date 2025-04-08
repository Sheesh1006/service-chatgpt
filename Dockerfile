# syntax=docker/dockerfile:1.2
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies including Git, gcc, etc.
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Use BuildKit secret to temporarily access the GitHub token
# Here we read the token from the mounted secret and modify the requirements.txt accordingly.
# Adjust the sed command if you need to target specific lines or repositories.
RUN --mount=type=secret,id=github_token \
    sh -c 'GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
           sed -i "s#git+https://github.com#git+https://${GITHUB_TOKEN}:@github.com#g" requirements.txt && \
           pip install --upgrade pip && \
           pip install --no-cache-dir -r requirements.txt'

# Copy the rest of your application code
COPY . .

EXPOSE 50051

CMD ["python", "main.py"]
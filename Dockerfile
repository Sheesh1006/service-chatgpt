FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Copy .netrc containing GitHub token into place (used by git clone)
COPY .netrc /root/.netrc

# Optional: secure .netrc
RUN chmod 600 /root/.netrc

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm /root/.netrc  # remove the token after use

COPY . .

EXPOSE 50051

CMD ["python", "main.py"]
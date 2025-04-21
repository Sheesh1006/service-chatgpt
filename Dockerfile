# Базовый образ
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка переменных окружения
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt
COPY requirements.txt .

# Замена placeholder на реальный токен и установка зависимостей
#ARG GITHUB_TOKEN
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование остального кода
COPY . .

# Команда для запуска приложения
CMD ["python", "main.py"]

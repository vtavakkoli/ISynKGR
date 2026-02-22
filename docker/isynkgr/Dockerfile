FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system appuser \
    && useradd --system --create-home --gid appuser --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir . \
    && chown -R appuser:appuser /app

USER appuser

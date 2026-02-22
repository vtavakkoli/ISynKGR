FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system appuser \
    && useradd --system --create-home --gid appuser --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

COPY . /app
RUN chown -R appuser:appuser /app

USER appuser

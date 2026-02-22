from __future__ import annotations

import logging
import uuid


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return logging.getLogger(name)


def correlation_id() -> str:
    return uuid.uuid4().hex[:12]

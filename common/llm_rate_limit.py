"""Bound retries to HTTP 429 and honor the provider's Retry-After."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

logger = logging.getLogger(__name__)
_MAX_WAIT_SECONDS = 120


def retry_delay(headers: httpx.Headers, attempt: int, now: datetime | None = None) -> float | None:
    value = headers.get("Retry-After")
    delay = None
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                delay = (when - (now or datetime.now(timezone.utc))).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
    if delay is not None and math.isfinite(delay):
        return max(5.0, delay) if delay <= _MAX_WAIT_SECONDS else None
    return min(_MAX_WAIT_SECONDS, 15.0 * 2**attempt)


class RateLimitedAsyncClient(httpx.AsyncClient):
    def __init__(self, *, rate_limit_retries: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.rate_limit_retries = max(0, min(3, rate_limit_retries))

    async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        for attempt in range(self.rate_limit_retries + 1):
            response = await super().send(request, **kwargs)
            if response.status_code != 429 or attempt == self.rate_limit_retries:
                return response
            delay = retry_delay(response.headers, attempt)
            if delay is None:
                return response
            await response.aclose()
            logger.warning("LLM HTTP 429, retry %d in %.1fs", attempt + 1, delay)
            await asyncio.sleep(delay)
        return response


class RateLimitedClient(httpx.Client):
    def __init__(self, *, rate_limit_retries: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.rate_limit_retries = max(0, min(3, rate_limit_retries))

    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        for attempt in range(self.rate_limit_retries + 1):
            response = super().send(request, **kwargs)
            if response.status_code != 429 or attempt == self.rate_limit_retries:
                return response
            delay = retry_delay(response.headers, attempt)
            if delay is None:
                return response
            response.close()
            logger.warning("LLM HTTP 429, retry %d in %.1fs", attempt + 1, delay)
            time.sleep(delay)
        return response


def chat_openai(*, rate_limit_retries: int = 2, **kwargs):
    from langchain_openai import ChatOpenAI

    timeout = kwargs.get("timeout", 120)
    return ChatOpenAI(
        http_client=RateLimitedClient(timeout=timeout, rate_limit_retries=rate_limit_retries),
        http_async_client=RateLimitedAsyncClient(timeout=timeout, rate_limit_retries=rate_limit_retries),
        max_retries=0,
        **kwargs,
    )

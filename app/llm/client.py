import asyncio
import random
import time
from dataclasses import dataclass, field

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.config import Settings
from app.llm.cache import ResponseCache

class EmptyResponseError(RuntimeError):
    """模型返回空内容（多为瞬时过载/思考耗尽），按可重试处理，避免缓存空响应。"""


RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, EmptyResponseError)


@dataclass
class LLMResult:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    cached: bool = False
    latency_ms: float = 0.0


class LLMClient:
    def __init__(self, settings: Settings, cache: ResponseCache | None = None):
        self.settings = settings
        self.cache = (
            cache
            if cache is not None
            else ResponseCache(settings.llm_cache_dir, settings.llm_cache_mode)
        )
        self._clients: dict[str, AsyncOpenAI] = {}
        self._sem = asyncio.Semaphore(settings.llm_max_concurrency)
        self._pace_lock = asyncio.Lock()
        self._last_request = 0.0

    def _provider_client(self, provider: str) -> AsyncOpenAI:
        if provider not in self._clients:
            base_url, api_key = self.settings.provider_credentials(provider)
            self._clients[provider] = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "missing-key",
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
            )
        return self._clients[provider]

    async def chat(
        self,
        role: str,
        messages: list,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra_body: dict | None = None,
        model_override: str | None = None,
    ) -> LLMResult:
        if model_override:
            provider, _, model = model_override.partition(":")
        else:
            provider, model = self.settings.role_model(role)
        params = {"temperature": temperature, "max_tokens": max_tokens, "extra_body": extra_body}
        key = self.cache.key(provider, model, messages, params)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResult(
                content=hit["content"], model=hit["model"], usage=hit.get("usage", {}), cached=True
            )
        started = time.perf_counter()
        async with self._sem:
            content, usage = await self._call_with_retry(provider, model, messages, params)
        latency_ms = (time.perf_counter() - started) * 1000
        self.cache.put(key, content, usage, model)
        return LLMResult(content=content, model=model, usage=usage, latency_ms=latency_ms)

    async def _call_with_retry(self, provider, model, messages, params) -> tuple[str, dict]:
        attempts = self.settings.llm_max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self._call_once(provider, model, messages, params)
            except RETRYABLE as exc:
                last_exc = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise
                last_exc = exc
            if attempt < attempts - 1:
                if isinstance(last_exc, RateLimitError):
                    # 免费档 1302 是账户级冷却，短退避无效，用长退避
                    delay = min(self.settings.llm_rate_limit_backoff_seconds * (2**attempt), 120.0)
                    if last_exc.response is not None:
                        retry_after = last_exc.response.headers.get("retry-after")
                        if retry_after:
                            try:
                                delay = max(delay, float(retry_after))
                            except ValueError:
                                pass
                else:
                    delay = min(2**attempt, 30)
                await asyncio.sleep(delay + random.uniform(0, 0.5))
        raise last_exc

    async def _call_once(self, provider, model, messages, params) -> tuple[str, dict]:
        async with self._pace_lock:
            wait = self.settings.llm_min_interval_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()
        extra_body = params.get("extra_body")
        call_params = {k: v for k, v in params.items() if k != "extra_body"}
        resp = await self._provider_client(provider).chat.completions.create(
            model=model, messages=messages, extra_body=extra_body, **call_params
        )
        usage = resp.usage.model_dump() if resp.usage else {}
        content = resp.choices[0].message.content or ""
        if not content.strip():
            raise EmptyResponseError(f"empty response from {model}")
        return content, usage
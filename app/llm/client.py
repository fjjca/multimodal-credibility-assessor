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

from app import obs
from app.config import Settings
from app.llm.cache import ResponseCache

class EmptyResponseError(RuntimeError):
    """模型返回空内容（多为瞬时过载/思考耗尽），按可重试处理，避免缓存空响应。"""


class CircuitOpenError(RuntimeError):
    """模型连续限流已被熔断，本轮直接跳过该候选（交由 fallback 链处理）。"""


@dataclass
class _Breaker:
    """单模型的限流熔断状态（进程内）。"""

    fails: int = 0
    opened_until: float = 0.0


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
        self._breakers: dict[str, _Breaker] = {}

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
        params = {"temperature": temperature, "max_tokens": max_tokens, "extra_body": extra_body}
        candidates = self._candidates(role, model_override)
        last_exc: Exception | None = None
        for idx, candidate in enumerate(candidates):
            provider, model = self.settings.split_candidate(candidate)
            if self._circuit_open(candidate):
                obs.record_llm_circuit_open(role, candidate)
                last_exc = CircuitOpenError(f"{candidate} 处于限流熔断冷却期")
                continue
            key = self.cache.key(provider, model, messages, params)
            hit = self.cache.get(key)
            if hit is not None:
                obs.record_llm_call(role, hit["model"], cached=True)
                return LLMResult(
                    content=hit["content"], model=hit["model"], usage=hit.get("usage", {}), cached=True
                )
            started = time.perf_counter()
            async with self._sem:
                try:
                    content, usage = await self._call_with_retry(
                        role, provider, model, messages, params
                    )
                except (RateLimitError, CircuitOpenError) as exc:
                    last_exc = exc
                    if idx < len(candidates) - 1:
                        obs.record_llm_fallback(role, candidate, candidates[idx + 1])
                        obs.log.warning(
                            "llm fallback",
                            extra={
                                "role": role, "from": candidate, "to": candidates[idx + 1],
                                "error": type(exc).__name__,
                            },
                        )
                        continue
                    raise
            latency_ms = (time.perf_counter() - started) * 1000
            self.cache.put(key, content, usage, model)
            obs.record_llm_call(
                role, model, cached=False,
                tokens=(usage or {}).get("total_tokens", 0), latency_ms=latency_ms,
            )
            return LLMResult(content=content, model=model, usage=usage, latency_ms=latency_ms)
        raise last_exc or RuntimeError("no available model for role " + role)

    def _candidates(self, role: str, model_override: str | None) -> list[str]:
        """候选模型链：主模型 + 备胎（限流/熔断时依次降级）。"""
        chain = self.settings.role_chain(role)
        if not model_override:
            return chain
        override = model_override.strip()
        # 指定模型（强裁判）时只向备胎降级，不退回主模型，避免"升级"名不副实
        out = [override]
        for cand in chain[1:]:
            if cand not in out:
                out.append(cand)
        return out

    def _note_rate_limit(self, candidate: str) -> bool:
        """记录一次限流；达到阈值则熔断并返回 True。"""
        breaker = self._breakers.setdefault(candidate, _Breaker())
        breaker.fails += 1
        if breaker.fails >= self.settings.llm_circuit_threshold:
            breaker.opened_until = time.monotonic() + self.settings.llm_circuit_cooldown_seconds
            return True
        return False

    def _circuit_open(self, candidate: str) -> bool:
        breaker = self._breakers.get(candidate)
        if breaker is None:
            return False
        if time.monotonic() < breaker.opened_until:
            return True
        if breaker.opened_until:
            # 冷却刚结束，重置计数重新统计
            breaker.opened_until = 0.0
            breaker.fails = 0
        return False

    def _rate_limit_delay(self, attempt: int, exc: RateLimitError) -> float:
        cap = self.settings.llm_rate_limit_backoff_max_seconds
        delay = min(self.settings.llm_rate_limit_backoff_seconds * (2**attempt), cap)
        if exc.response is not None:
            retry_after = exc.response.headers.get("retry-after")
            if retry_after:
                try:
                    delay = min(max(delay, float(retry_after)), cap)
                except ValueError:
                    pass
        return delay

    async def _call_with_retry(self, role, provider, model, messages, params) -> tuple[str, dict]:
        candidate = f"{provider}:{model}"
        attempts = self.settings.retry_budget(role) + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self._call_once(provider, model, messages, params)
            except RateLimitError as exc:
                last_exc = exc
                obs.record_llm_rate_limited(role, candidate)
                if self._note_rate_limit(candidate):
                    obs.log.warning(
                        "llm circuit open",
                        extra={"role": role, "model": candidate, "fails": self.settings.llm_circuit_threshold},
                    )
                    raise
                if attempt >= attempts - 1:
                    break
                delay = self._rate_limit_delay(attempt, exc)
                obs.log.warning(
                    "llm retry",
                    extra={
                        "role": role, "model": candidate, "attempt": attempt + 1,
                        "error": type(exc).__name__, "delay_s": round(delay, 2),
                    },
                )
                await asyncio.sleep(delay + random.uniform(0, 0.5))
            except (APITimeoutError, APIConnectionError, EmptyResponseError) as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    break
                delay = min(2**attempt, 30)
                obs.log.warning(
                    "llm retry",
                    extra={
                        "role": role, "model": candidate, "attempt": attempt + 1,
                        "error": type(exc).__name__, "delay_s": round(delay, 2),
                    },
                )
                await asyncio.sleep(delay + random.uniform(0, 0.5))
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise
                last_exc = exc
                if attempt >= attempts - 1:
                    break
                delay = min(2**attempt, 30)
                obs.log.warning(
                    "llm retry",
                    extra={
                        "role": role, "model": candidate, "attempt": attempt + 1,
                        "error": type(exc).__name__, "delay_s": round(delay, 2),
                    },
                )
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
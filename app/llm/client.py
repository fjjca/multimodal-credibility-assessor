import asyncio
import json
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


# 兜底候选链的触发条件：限流/熔断，以及重试耗尽后的超时/连接失败/空响应/5xx。
# 4xx（请求本身有问题）一般不触发兜底——换模型同样会失败，只会白白多烧一次配额；
# 例外是「该模型不支持某参数」（见 unsupported_param_detail），那是模型自身限制，换备胎可能成功。
FALLBACK_TRIGGER = (
    RateLimitError, CircuitOpenError, APITimeoutError, APIConnectionError, EmptyResponseError,
)

# 部分模型不支持的参数（智谱/硅基流动的视觉模型对 enable_thinking 返回 400/20015）
_THINKING_KEYS = ("thinking", "enable_thinking")


def strip_thinking(body: dict | None) -> dict | None:
    """剔除 thinking 相关字段；没有可剔除的字段时原样返回（同一对象）。"""
    if not body:
        return body
    out = {k: v for k, v in body.items() if k not in _THINKING_KEYS}
    return out if out != body else body


def error_body(exc: Exception) -> dict:
    """提取 API 错误的 JSON body（没有就返回空 dict）。"""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError):
            return {"raw": body[:200]}
        return parsed if isinstance(parsed, dict) else {"raw": body[:200]}
    return {}


def unsupported_param_detail(exc: Exception) -> str:
    """判断错误是否为「模型不支持某参数」，返回可读说明；不是则返回空串。"""
    if not isinstance(exc, APIStatusError) or exc.status_code != 400:
        return ""
    body = error_body(exc)
    code = str(body.get("code", ""))
    msg = str(body.get("message") or body.get("raw") or "")
    if code == "20015" or "does not support parameter" in msg or "unsupported parameter" in msg:
        return (msg or f"code={code}")[:160]
    return ""


# 内容审核拒答的判定：智谱审核类错误码 + 消息关键词。参数错误（如 400/20015）不属于审核，
# 早期把所有 400 都写成「内容审核拒答」，把参数问题伪装成了审核问题，误导了排查方向。
MODERATION_CODES = {"1113", "1211", "1225", "1301", "1502"}
MODERATION_KEYWORDS = ("违规", "审核", "敏感", "safety", "content_policy", "policy", "risk")


def is_moderation_error(exc: Exception) -> bool:
    """该 400 是否为内容审核拒答（而非请求参数/其它问题）。"""
    body = error_body(exc)
    if str(body.get("code", "")) in MODERATION_CODES:
        return True
    msg = str(body.get("message") or body.get("raw") or "").lower()
    return any(k in msg for k in MODERATION_KEYWORDS)


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
        # 不支持 thinking 参数的候选模型：配置里声明的 + 运行时遇到 400/20015 学到的
        self._no_thinking: set[str] = set(settings.no_thinking_models())

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
        base_params = {"temperature": temperature, "max_tokens": max_tokens, "extra_body": extra_body}
        candidates = self._candidates(role, model_override)
        last_exc: Exception | None = None
        for idx, candidate in enumerate(candidates):
            provider, model = self.settings.split_candidate(candidate)
            if self._circuit_open(candidate):
                obs.record_llm_circuit_open(role, candidate)
                last_exc = CircuitOpenError(f"{candidate} 处于限流熔断冷却期")
                continue
            params = self._params_for(candidate, base_params)
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
                    content, usage, eff_params = await self._attempt(
                        role, candidate, provider, model, messages, params
                    )
                except (FALLBACK_TRIGGER + (APIStatusError,)) as exc:
                    last_exc = exc
                    if idx < len(candidates) - 1 and self._fallback_worthy(exc):
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
            if eff_params is not params:
                key = self.cache.key(provider, model, messages, eff_params)
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

    @staticmethod
    def _fallback_worthy(exc: Exception) -> bool:
        """该异常是否值得换备胎：429/熔断/超时/连接失败/空响应/5xx 值得；一般 4xx 不值得，
        但「模型不支持某参数」是模型自身限制，同一请求对备胎可能合法，值得一试。"""
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APIStatusError):
            if exc.status_code >= 500:
                return True
            if exc.status_code == 400 and unsupported_param_detail(exc):
                return True
            return False
        return True

    def _params_for(self, candidate: str, params: dict) -> dict:
        """按候选模型裁剪请求参数：不支持 thinking 的模型直接剔除 thinking 字段。"""
        body = params.get("extra_body")
        if not body or candidate not in self._no_thinking:
            return params
        stripped = strip_thinking(body)
        if stripped is body:
            return params
        return {**params, "extra_body": stripped or None}

    async def _attempt(
        self, role: str, candidate: str, provider: str, model: str, messages: list, params: dict
    ) -> tuple[str, dict, dict]:
        """调用候选模型；若因模型不支持 thinking 类参数被 400 拒绝，剔除参数后原地重试一次。"""
        try:
            content, usage = await self._call_with_retry(role, provider, model, messages, params)
            return content, usage, params
        except APIStatusError as exc:
            detail = unsupported_param_detail(exc)
            body = params.get("extra_body")
            stripped = strip_thinking(body) if body else None
            if not detail or stripped is None or stripped is body:
                raise
            self._no_thinking.add(candidate)
            obs.record_llm_param_stripped(role, candidate)
            obs.log.warning(
                "llm unsupported param, retry without thinking",
                extra={"role": role, "model": candidate, "detail": detail},
            )
            new_params = {**params, "extra_body": stripped or None}
            content, usage = await self._call_with_retry(role, provider, model, messages, new_params)
            return content, usage, new_params

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
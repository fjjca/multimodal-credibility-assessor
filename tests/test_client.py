import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app import obs
from app.config import Settings
from app.llm.cache import ReplayMissError
from app.llm.client import CircuitOpenError, EmptyResponseError, LLMClient


def _rate_limit(retry_after: str | None = None):
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(429, request=httpx.Request("POST", "http://test"), headers=headers)
    return RateLimitError("429 rate limited", response=resp, body=None)


def _timeout():
    return APITimeoutError(request=httpx.Request("POST", "http://test"))


def _status_error(code: int):
    resp = httpx.Response(code, request=httpx.Request("POST", "http://test"))
    return APIStatusError(f"http {code}", response=resp, body=None)


def make_client(tmp_path, **overrides) -> LLMClient:
    settings = Settings(_env_file=None, llm_cache_dir=str(tmp_path), **overrides)
    return LLMClient(settings)


async def test_cache_hit_skips_network(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    calls = []

    async def fake_call_once(provider, model, messages, params):
        calls.append(1)
        return "ok", {"total_tokens": 5}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    msgs = [{"role": "user", "content": "hi"}]
    r1 = await client.chat("decompose", msgs)
    r2 = await client.chat("decompose", msgs)
    assert r1.content == "ok" and not r1.cached
    assert r2.cached and r2.content == "ok"
    assert len(calls) == 1


async def test_retry_then_success(tmp_path, monkeypatch):
    client = make_client(tmp_path, llm_max_retries=2)
    attempts = []

    async def flaky(provider, model, messages, params):
        attempts.append(1)
        if len(attempts) < 3:
            raise APIConnectionError(request=httpx.Request("POST", "http://test"))
        return "ok", {}

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(client, "_call_once", flaky)
    monkeypatch.setattr("app.llm.client.asyncio.sleep", no_sleep)
    result = await client.chat("judge", [{"role": "user", "content": "x"}])
    assert result.content == "ok"
    assert len(attempts) == 3


async def test_replay_mode_miss_raises(tmp_path):
    client = make_client(tmp_path, llm_cache_mode="replay")
    with pytest.raises(ReplayMissError):
        await client.chat("decompose", [{"role": "user", "content": "never recorded"}])


async def test_empty_response_retried_then_cached(tmp_path, monkeypatch):
    client = make_client(tmp_path, llm_max_retries=2)
    attempts = []

    async def flaky(provider, model, messages, params):
        attempts.append(1)
        if len(attempts) < 3:
            raise EmptyResponseError("empty")
        return "ok", {}

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(client, "_call_once", flaky)
    monkeypatch.setattr("app.llm.client.asyncio.sleep", no_sleep)
    msgs = [{"role": "user", "content": "x"}]
    r = await client.chat("judge", msgs)
    assert r.content == "ok"
    assert len(attempts) == 3
    r2 = await client.chat("judge", msgs)
    assert r2.cached  # 非空结果正常缓存


async def test_empty_response_not_cached(tmp_path, monkeypatch):
    client = make_client(tmp_path, llm_max_retries=1)

    async def always_empty(provider, model, messages, params):
        raise EmptyResponseError("empty")

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(client, "_call_once", always_empty)
    monkeypatch.setattr("app.llm.client.asyncio.sleep", no_sleep)
    with pytest.raises(EmptyResponseError):
        await client.chat("judge", [{"role": "user", "content": "x"}])
    assert list(tmp_path.glob("*.json")) == []  # 空响应未写入缓存


async def test_call_once_detects_empty_content(tmp_path, monkeypatch):
    client = make_client(tmp_path, llm_max_retries=0)

    class FakeMsg:
        content = ""

    class FakeChoice:
        message = FakeMsg()

    class FakeResp:
        choices = [FakeChoice()]
        usage = None

    class FakeCompletions:
        async def create(self, **kw):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeProvider:
        chat = FakeChat()

    monkeypatch.setattr(client, "_provider_client", lambda provider: FakeProvider())
    with pytest.raises(EmptyResponseError):
        await client.chat("judge", [{"role": "user", "content": "x"}])

async def test_fallback_chain_switches_model_on_rate_limit(tmp_path, monkeypatch):
    client = make_client(
        tmp_path,
        llm_max_retries=0,
        llm_role_fallbacks={"decompose": ["siliconflow:Qwen/Qwen3-8B"]},
    )
    calls = []

    async def fake_call_once(provider, model, messages, params):
        calls.append(f"{provider}:{model}")
        if provider == "zhipu":
            raise _rate_limit()
        return "ok", {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    r = await client.chat("decompose", [{"role": "user", "content": "hi"}])
    assert r.content == "ok"
    assert calls == ["zhipu:glm-4-flash", "siliconflow:Qwen/Qwen3-8B"]
    # 兜底切换必须可观测（dashboard 的"限流与降级"面板依赖这两个计数器）
    assert obs.LLM_RATE_LIMITED.labels(role="decompose", model="zhipu:glm-4-flash")._value.get() >= 1
    assert obs.LLM_FALLBACKS.labels(
        role="decompose",
        from_model="zhipu:glm-4-flash",
        to_model="siliconflow:Qwen/Qwen3-8B",
    )._value.get() >= 1


async def test_circuit_breaker_skips_model_after_repeated_429(tmp_path, monkeypatch):
    client = make_client(tmp_path, llm_max_retries=0, llm_circuit_threshold=2)
    calls = []

    async def fake_call_once(provider, model, messages, params):
        calls.append(model)
        raise _rate_limit()

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(RateLimitError):
        await client.chat("judge", msgs)
    with pytest.raises(RateLimitError):
        await client.chat("judge", msgs)   # 第二次达到阈值，熔断打开
    with pytest.raises(CircuitOpenError):
        await client.chat("judge", msgs)   # 冷却期内直接跳过，不再打 API
    assert len(calls) == 2
    assert obs.LLM_CIRCUIT_OPEN.labels(role="judge", model="zhipu:glm-4.7-flash")._value.get() >= 1


async def test_retry_budget_per_role(tmp_path, monkeypatch):
    client = make_client(tmp_path, llm_retry_budgets={"judge": 1}, llm_max_retries=5)
    attempts = []

    async def fake_call_once(provider, model, messages, params):
        attempts.append(1)
        raise _rate_limit()

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    monkeypatch.setattr("app.llm.client.asyncio.sleep", no_sleep)
    with pytest.raises(RateLimitError):
        await client.chat("judge", [{"role": "user", "content": "x"}])
    assert len(attempts) == 2      # budget=1 → 总共 2 次尝试


async def test_rate_limit_backoff_capped(tmp_path, monkeypatch):
    client = make_client(
        tmp_path,
        llm_max_retries=1,
        llm_min_interval_seconds=0.0,
        llm_rate_limit_backoff_seconds=15.0,
        llm_rate_limit_backoff_max_seconds=20.0,
    )
    sleeps = []

    async def fake_call_once(provider, model, messages, params):
        raise _rate_limit(retry_after="600")

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    monkeypatch.setattr("app.llm.client.asyncio.sleep", record_sleep)
    with pytest.raises(RateLimitError):
        await client.chat("judge", [{"role": "user", "content": "x"}])
    assert sleeps and max(sleeps) <= 20.5   # retry-after=600 也被上限截断


def test_role_chain_skips_invalid_fallbacks():
    s = Settings(
        _env_file=None,
        llm_role_fallbacks={
            "decompose": ["", "badprovider:x", "zhipu:glm-4.7-flashx", "zhipu:glm-4-flash"],
        },
    )
    assert s.role_chain("decompose") == ["zhipu:glm-4-flash", "zhipu:glm-4.7-flashx"]


def test_settings_parses_fallbacks_from_env(monkeypatch):
    monkeypatch.setenv(
        "LLM_ROLE_FALLBACKS", '{"forensics":["siliconflow:PaddlePaddle/PaddleOCR-VL-1.5"]}'
    )
    s = Settings(_env_file=None)
    assert s.role_chain("forensics") == [
        "zhipu:glm-4.6v-flash",
        "siliconflow:PaddlePaddle/PaddleOCR-VL-1.5",
    ]


def test_override_chain_dedupes_and_never_falls_back_to_primary():
    client = LLMClient(
        Settings(
            _env_file=None,
            llm_role_fallbacks={"judge": ["zhipu:glm-4.5-flash", "siliconflow:Qwen/Qwen3-8B"]},
        )
    )
    # 强裁判与 judge 的备胎是同一个模型：不应重复尝试
    assert client._candidates("judge", "zhipu:glm-4.5-flash") == [
        "zhipu:glm-4.5-flash",
        "siliconflow:Qwen/Qwen3-8B",
    ]
    # 升级不得退回主模型
    assert client._candidates("judge", "zhipu:glm-4.9-flash")[0] == "zhipu:glm-4.9-flash"
    assert "zhipu:glm-4.7-flash" not in client._candidates("judge", "zhipu:glm-4.9-flash")


async def test_fallback_chain_switches_on_timeout(tmp_path, monkeypatch):
    """重试耗尽的超时也应切备胎——免费档"卡住"和 429 一样致命。"""
    client = make_client(
        tmp_path,
        llm_max_retries=0,
        llm_role_fallbacks={"judge": ["siliconflow:zai-org/GLM-4.5-Air"]},
    )
    calls = []

    async def fake_call_once(provider, model, messages, params):
        calls.append(f"{provider}:{model}")
        if provider == "zhipu":
            raise _timeout()
        return "ok", {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    r = await client.chat("judge", [{"role": "user", "content": "hi"}])
    assert r.content == "ok"
    assert calls == ["zhipu:glm-4.7-flash", "siliconflow:zai-org/GLM-4.5-Air"]
    assert obs.LLM_FALLBACKS.labels(
        role="judge",
        from_model="zhipu:glm-4.7-flash",
        to_model="siliconflow:zai-org/GLM-4.5-Air",
    )._value.get() >= 1


async def test_fallback_chain_switches_on_server_error(tmp_path, monkeypatch):
    client = make_client(
        tmp_path,
        llm_max_retries=0,
        llm_role_fallbacks={"judge": ["siliconflow:zai-org/GLM-4.5-Air"]},
    )
    calls = []

    async def fake_call_once(provider, model, messages, params):
        calls.append(f"{provider}:{model}")
        if provider == "zhipu":
            raise _status_error(500)
        return "ok", {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    r = await client.chat("judge", [{"role": "user", "content": "hi"}])
    assert r.content == "ok"
    assert calls == ["zhipu:glm-4.7-flash", "siliconflow:zai-org/GLM-4.5-Air"]


async def test_no_fallback_on_client_error(tmp_path, monkeypatch):
    """4xx 是请求本身的问题，换模型同样会失败，不该白烧一次配额。"""
    client = make_client(
        tmp_path,
        llm_max_retries=0,
        llm_role_fallbacks={"judge": ["siliconflow:zai-org/GLM-4.5-Air"]},
    )
    calls = []

    async def fake_call_once(provider, model, messages, params):
        calls.append(model)
        raise _status_error(400)

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    with pytest.raises(APIStatusError):
        await client.chat("judge", [{"role": "user", "content": "hi"}])
    assert calls == ["glm-4.7-flash"]


def _param_error():
    resp = httpx.Response(400, request=httpx.Request("POST", "http://test"))
    body = {
        "code": 20015,
        "message": "Value error, current model does not support parameter `enable_thinking`.",
    }
    return APIStatusError("http 400", response=resp, body=body)


async def test_thinking_stripped_for_models_that_reject_it(tmp_path, monkeypatch):
    """视觉模型对 enable_thinking 返回 400/20015，客户端按候选模型自动剔除该参数。"""
    client = make_client(tmp_path, llm_max_retries=0)
    seen = []

    async def fake_call_once(provider, model, messages, params):
        seen.append(params.get("extra_body"))
        return "ok", {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    body = {"thinking": {"type": "disabled"}}
    await client.chat("forensics", [{"role": "user", "content": "hi"}], extra_body=body)
    assert seen == [None]   # 主模型 zhipu:glm-4.6v-flash 在 no-thinking 名单里


async def test_thinking_kept_for_supported_models(tmp_path, monkeypatch):
    client = make_client(tmp_path, llm_max_retries=0)
    seen = []

    async def fake_call_once(provider, model, messages, params):
        seen.append(params.get("extra_body"))
        return "ok", {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    body = {"thinking": {"type": "disabled"}}
    await client.chat("decompose", [{"role": "user", "content": "hi"}], extra_body=body)
    assert seen == [body]   # 文本模型照常下发


async def test_unsupported_param_triggers_retry_without_thinking(tmp_path, monkeypatch):
    """未知模型报 400/20015 时，剔除 thinking 原地重试一次，而不是直接失败。"""
    client = make_client(tmp_path, llm_max_retries=0)
    seen = []

    async def fake_call_once(provider, model, messages, params):
        seen.append(params.get("extra_body"))
        if len(seen) == 1:
            raise _param_error()
        return "ok", {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    body = {"thinking": {"type": "disabled"}}
    r = await client.chat("decompose", [{"role": "user", "content": "hi"}], extra_body=body)
    assert r.content == "ok"
    assert seen == [body, None]
    assert obs.LLM_PARAM_STRIPPED.labels(
        role="decompose", model="zhipu:glm-4-flash"
    )._value.get() >= 1


async def test_param_error_falls_back_to_backup_model(tmp_path, monkeypatch):
    """参数错误是模型自身限制，同一请求对备胎可能合法——允许切备胎。"""
    client = make_client(
        tmp_path,
        llm_max_retries=0,
        llm_role_fallbacks={"decompose": ["siliconflow:Qwen/Qwen3-8B"]},
    )
    calls = []

    async def fake_call_once(provider, model, messages, params):
        calls.append(f"{provider}:{model}")
        if provider == "zhipu":
            raise _param_error()
        return "ok", {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    r = await client.chat("decompose", [{"role": "user", "content": "hi"}],
                          extra_body={"thinking": {"type": "disabled"}})
    assert r.content == "ok"
    # 主模型自愈重试一次（不带 thinking）后仍失败，才切备胎
    assert calls == ["zhipu:glm-4-flash", "zhipu:glm-4-flash", "siliconflow:Qwen/Qwen3-8B"]

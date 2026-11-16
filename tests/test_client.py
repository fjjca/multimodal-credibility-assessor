import httpx
import pytest
from openai import APIConnectionError

from app.config import Settings
from app.llm.cache import ReplayMissError
from app.llm.client import EmptyResponseError, LLMClient


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
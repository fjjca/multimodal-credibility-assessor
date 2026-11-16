import pytest

from app.llm.cache import ReplayMissError, ResponseCache


def test_key_stable_and_sensitive():
    msgs = [{"role": "user", "content": "hi"}]
    k1 = ResponseCache.key("zhipu", "m", msgs, {"temperature": 0.2})
    k2 = ResponseCache.key("zhipu", "m", msgs, {"temperature": 0.2})
    k3 = ResponseCache.key("zhipu", "m", msgs, {"temperature": 0.9})
    assert k1 == k2
    assert k1 != k3


def test_roundtrip(tmp_path):
    cache = ResponseCache(str(tmp_path), "readwrite")
    assert cache.get("k") is None
    cache.put("k", "hello", {"total_tokens": 3}, "m")
    hit = cache.get("k")
    assert hit["content"] == "hello"
    assert hit["usage"]["total_tokens"] == 3


def test_replay_miss(tmp_path):
    cache = ResponseCache(str(tmp_path), "replay")
    with pytest.raises(ReplayMissError):
        cache.get("absent")


def test_off_mode_never_stores(tmp_path):
    cache = ResponseCache(str(tmp_path), "off")
    cache.put("k", "x", {}, "m")
    assert cache.get("k") is None
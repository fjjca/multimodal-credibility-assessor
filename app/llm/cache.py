import hashlib
import json
import time
from pathlib import Path


class ReplayMissError(RuntimeError):
    pass


class ResponseCache:
    """按 (provider, model, messages, params) 哈希缓存响应。

    replay 模式只读已录制的响应，未命中抛错，用于无 key 离线开发。
    """

    def __init__(self, directory: str, mode: str = "readwrite"):
        if mode not in ("off", "readwrite", "replay"):
            raise ValueError(f"invalid cache mode: {mode}")
        self.mode = mode
        self.dir = Path(directory)
        if mode != "off":
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(provider: str, model: str, messages: list, params: dict) -> str:
        payload = json.dumps(
            {"provider": provider, "model": model, "messages": messages, "params": params},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        if self.mode == "off":
            return None
        path = self.dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.mode == "replay":
            raise ReplayMissError(f"no recorded response for key {key}")
        return None

    def put(self, key: str, content: str, usage: dict, model: str) -> None:
        if self.mode != "readwrite":
            return
        entry = {"content": content, "usage": usage, "model": model, "created_at": time.time()}
        (self.dir / f"{key}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
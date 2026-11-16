import json

import httpx
import pytest

from app.main import app


async def _fake_pipeline(text, image_name, emit, **kwargs):
    stages = [
        ("decompose", "断言拆解"),
        ("forensics", "多模态取证"),
        ("retrieve", "证据检索"),
        ("graph", "证据图构建"),
        ("adjudicate", "多模型裁决"),
    ]
    for sid, label in stages:
        await emit("stage", {"stage": sid, "label": label, "status": "running"})
        status = "skipped" if sid == "forensics" else "done"
        await emit("stage", {"stage": sid, "label": label, "status": status, "detail": {}})
    result = {"verdict": "unverifiable", "verdict_label": "无法验证", "claims": []}
    await emit("result", result)
    return result


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch):
    monkeypatch.setattr("app.api.run_pipeline", _fake_pipeline)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_verify_end_to_end(client):
    res = await client.post("/api/verify", data={"text": "测试帖子"})
    assert res.status_code == 200
    task_id = res.json()["task_id"]

    events = []
    async with client.stream("GET", f"/api/verify/{task_id}/events") as stream:
        event_name = None
        async for line in stream.aiter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                events.append((event_name, json.loads(line[len("data: "):])))

    names = [name for name, _ in events]
    assert names.count("stage") == 10
    assert names[-1] == "result"
    assert events[-1][1]["verdict_label"] == "无法验证"

    status = await client.get(f"/api/verify/{task_id}")
    assert status.json()["status"] == "done"


async def test_verify_with_image(client):
    res = await client.post(
        "/api/verify",
        data={"text": "带图帖子"},
        files={"image": ("../evil.png", b"fake-bytes", "image/png")},
    )
    assert res.status_code == 200
    task_id = res.json()["task_id"]
    status = await client.get(f"/api/verify/{task_id}")
    assert status.status_code == 200


async def test_empty_text_rejected(client):
    res = await client.post("/api/verify", data={"text": "   "})
    assert res.status_code == 422


async def test_unknown_task_404(client):
    res = await client.get("/api/verify/nonexistent/events")
    assert res.status_code == 404
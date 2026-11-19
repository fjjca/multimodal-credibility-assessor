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


async def test_list_evals(client, tmp_path, monkeypatch):
    run_dir = tmp_path / "abl_text_linear"
    run_dir.mkdir()
    summary = {
        "system": "pipeline", "dataset": "data/eval_seed.jsonl", "timestamp": 123.0,
        "n_failed": 0, "metrics": {"n": 6, "accuracy": 0.5, "ece": 0.32},
    }
    (run_dir / "eval_pipeline_20261118_180034.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    monkeypatch.setattr("app.api.EVAL_DIR", tmp_path)
    res = await client.get("/api/evals")
    assert res.status_code == 200
    items = res.json()["evals"]
    assert len(items) == 1
    assert items[0]["dir"] == "abl_text_linear"
    assert items[0]["metrics"]["accuracy"] == 0.5


async def test_list_evals_empty(client, tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.EVAL_DIR", tmp_path / "nope")
    res = await client.get("/api/evals")
    assert res.status_code == 200
    assert res.json()["evals"] == []


async def test_calibration_endpoint(client, tmp_path, monkeypatch):
    from app.config import Settings

    (tmp_path / "calibrator.json").write_text('{"temperature": 5.6}', encoding="utf-8")
    (tmp_path / "reliability.json").write_text(
        json.dumps({"temperature": 5.6, "n_attempted": 5, "reliability_raw": []}), encoding="utf-8"
    )
    settings = Settings(_env_file=None, calibrator_path=str(tmp_path / "calibrator.json"))
    monkeypatch.setattr("app.api.get_settings", lambda: settings)
    res = await client.get("/api/calibration")
    assert res.status_code == 200
    body = res.json()
    assert body["calibrator"]["temperature"] == 5.6
    assert body["reliability"]["n_attempted"] == 5


async def test_calibration_missing(client, tmp_path, monkeypatch):
    from app.config import Settings

    settings = Settings(_env_file=None, calibrator_path=str(tmp_path / "absent.json"))
    monkeypatch.setattr("app.api.get_settings", lambda: settings)
    res = await client.get("/api/calibration")
    assert res.status_code == 200
    assert res.json() == {"calibrator": None, "reliability": None}
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import obs
from app.config import get_settings
from app.pipeline.runner import run_pipeline

router = APIRouter(prefix="/api")
UPLOAD_DIR = Path(".data/uploads")
EVAL_DIR = Path("out")


@dataclass
class Task:
    id: str
    text: str
    image_path: str | None
    events: list = field(default_factory=list)
    status: str = "running"
    result: dict | None = None
    notify: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False


_tasks: dict[str, Task] = {}


def _make_emit(task: Task):
    async def emit(event: str, data: dict) -> None:
        task.events.append({"event": event, "data": data, "ts": time.time()})
        task.notify.set()

    return emit


async def _run(task: Task) -> None:
    emit = _make_emit(task)
    started = time.perf_counter()
    try:
        task.result = await run_pipeline(task.text, task.image_path, emit)
        task.status = "done"
        elapsed = time.perf_counter() - started
        obs.record_verify(
            latency_s=elapsed,
            verdict=task.result.get("verdict", "unknown"),
            escalated=bool(task.result.get("escalated", False)),
        )
        obs.log.info(
            "verify done",
            extra={
                "task_id": task.id,
                "verdict": task.result.get("verdict"),
                "confidence": task.result.get("confidence"),
                "latency_s": round(elapsed, 3),
                "total_tokens": (task.result.get("usage") or {}).get("total_tokens", 0),
            },
        )
    except Exception as exc:
        task.status = "error"
        obs.VERIFY_ERRORS.inc()
        obs.log.error("verify failed", extra={"task_id": task.id, "error": str(exc)}, exc_info=True)
        await emit("fail", {"message": str(exc)})
    finally:
        task.finished = True
        task.notify.set()


@router.post("/verify")
async def verify(text: str = Form(...), image: UploadFile | None = File(None)):
    if not text.strip():
        raise HTTPException(status_code=422, detail="text 不能为空")
    task_id = uuid.uuid4().hex[:12]
    image_path = None
    if image is not None and image.filename:
        image_name = Path(image.filename).name
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / f"{task_id}_{image_name}"
        dest.write_bytes(await image.read())
        image_path = str(dest)
    task = Task(id=task_id, text=text, image_path=image_path)
    _tasks[task_id] = task
    obs.record_verify_request(has_image=image_path is not None)
    obs.log.info("verify accepted", extra={"task_id": task_id, "has_image": image_path is not None})
    asyncio.create_task(_run(task))
    return {"task_id": task_id}


@router.get("/verify/{task_id}")
async def get_task(task_id: str):
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {"id": task.id, "status": task.status, "result": task.result, "events": task.events}


@router.get("/verify/{task_id}/events")
async def task_events(task_id: str):
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")

    async def gen():
        i = 0
        while True:
            while i < len(task.events):
                p = task.events[i]
                i += 1
                yield f"event: {p['event']}\ndata: {json.dumps(p['data'], ensure_ascii=False)}\n\n"
            if task.finished:
                break
            task.notify.clear()
            if i < len(task.events) or task.finished:
                continue
            await task.notify.wait()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/evals")
async def list_evals():
    """扫描 out/ 下的评测汇总（只读本地 artifact，不接受用户路径，避免遍历）。供 dashboard 对比消融。"""
    items = []
    if EVAL_DIR.exists():
        for path in sorted(EVAL_DIR.glob("**/eval_*.json")):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            items.append(
                {
                    "dir": path.parent.name,
                    "file": path.name,
                    "system": summary.get("system"),
                    "dataset": summary.get("dataset"),
                    "timestamp": summary.get("timestamp"),
                    "n_failed": summary.get("n_failed", 0),
                    "metrics": summary.get("metrics", {}),
                }
            )
    items.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
    return {"evals": items}


@router.get("/calibration")
async def get_calibration():
    """返回校准器元数据与可靠性分箱（scripts.calibrate 落盘的 artifact）。缺失时各项为 null。"""
    settings = get_settings()
    cal_path = Path(settings.calibrator_path)
    rel_path = cal_path.parent / "reliability.json"

    def _read(p: Path):
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    return {"calibrator": _read(cal_path), "reliability": _read(rel_path)}
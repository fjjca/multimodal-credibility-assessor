import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.pipeline.runner import run_pipeline

router = APIRouter(prefix="/api")
UPLOAD_DIR = Path(".data/uploads")


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
    try:
        task.result = await run_pipeline(task.text, task.image_path, emit)
        task.status = "done"
    except Exception as exc:
        task.status = "error"
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
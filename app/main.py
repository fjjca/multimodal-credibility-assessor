from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from app import obs
from app.api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    obs.setup_logging()
    yield


app = FastAPI(title="多模态内容可信度核查", version="0.0.1", lifespan=lifespan)
app.include_router(router)


@app.get("/metrics", include_in_schema=False)
async def metrics():
    body, content_type = obs.metrics_payload()
    return Response(content=body, media_type=content_type)


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True),
    name="static",
)
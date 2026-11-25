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


@app.middleware("http")
async def no_cache_static(request, call_next):
    # 前端是单页 index.html，浏览器缓存它会让人以为改了没生效
    response = await call_next(request)
    ctype = response.headers.get("content-type", "")
    if "text/html" in ctype or "javascript" in ctype or "text/css" in ctype:
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/metrics", include_in_schema=False)
async def metrics():
    body, content_type = obs.metrics_payload()
    return Response(content=body, media_type=content_type)


_DEMO_PHOTOS = Path(__file__).resolve().parent.parent / "data" / "demo_photos"
_DEMO_PHOTOS.mkdir(parents=True, exist_ok=True)

app.mount(
    "/demo_photos",
    StaticFiles(directory=_DEMO_PHOTOS),
    name="demo_photos",
)

app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True),
    name="static",
)
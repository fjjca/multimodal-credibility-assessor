"""下载并归一化演示用的真实网络图片。

图片来自国内图片站点的公开 CDN，非开放许可，因此不入版本库（见 .gitignore）；
本脚本按 SOURCES.md 里记录的来源重新拉取，演示与评测可复现。

用法：
    .venv/Scripts/python -m scripts.fetch_demo_photos
"""

from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "demo_photos"

SOURCES = {
    "paris_eiffel.jpg": "http://b0.bdstatic.com/a6635005d706780b4b15385fb31ed72f.jpg",
    "beijing_marathon.jpg": (
        "http://q6.itc.cn/q_70/images03/20240414/c1c96323690d4ddf9cc3ae68c4e3f557.jpeg"
    ),
    "supermarket_receipt.jpg": "http://b0.bdstatic.com/b15f93965018b0c4723de74f422ebd52.jpg",
}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://www.baidu.com/",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        resp = requests.get(url, timeout=60, headers=HEADERS)
        resp.raise_for_status()
        im = Image.open(BytesIO(resp.content)).convert("RGB")
        im.thumbnail((1024, 1024), Image.LANCZOS)
        out = DEST / name
        im.save(out, "JPEG", quality=88)
        print(f"{name}: {im.size} -> {out}")


if __name__ == "__main__":
    main()

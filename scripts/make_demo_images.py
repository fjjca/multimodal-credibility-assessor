"""生成 M3 多模态合成演示/评测用例（免费、可复现）。

图像含可被 VLM OCR 识别的文字，与帖子文本构成一致/矛盾关系。
Fakeddit 真实图像因 Reddit 被墙不可用，VLM 也无法抓取境外 URL，故用本地图 base64。
运行: python -m scripts.make_demo_images
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_IMG = Path("data/demo_images")
OUT_JSONL = Path("data/multimodal_demo.jsonl")
FONT = "C:/Windows/Fonts/arial.ttf"

CASES = [
    {
        "id": "mm01", "label": "refuted",
        "lines": ["PARIS 2024", "EIFFEL TOWER", "OLYMPIC GAMES"],
        "text": "这张照片拍摄于2020年东京奥运会现场。",
    },
    {
        "id": "mm02", "label": "supported",
        "lines": ["WHO GUIDELINE", "150 minutes", "moderate exercise", "per week"],
        "text": "世卫组织建议成年人每周至少进行150分钟中等强度运动。",
    },
    {
        "id": "mm03", "label": "refuted",
        "lines": ["RECEIPT", "TOTAL: $128.00", "THANK YOU"],
        "text": "这张小票显示本次消费总额为500元。",
    },
    {
        "id": "mm04", "label": "supported",
        "lines": ["RECEIPT", "TOTAL: $128.00", "THANK YOU"],
        "text": "这张小票显示本次消费总额为128美元。",
    },
    {
        "id": "mm05", "label": "refuted",
        "lines": ["NEWS", "MARATHON 2023", "CITY RUNNING EVENT"],
        "text": "这张新闻截图证实本市马拉松于2024年举办。",
    },
    {
        "id": "mm06", "label": "supported",
        "lines": ["SPEED LIMIT", "60", "km/h"],
        "text": "这张交通标志照片显示该路段限速60公里每小时。",
    },
]


def render(lines: list[str], path: Path) -> None:
    img = Image.new("RGB", (640, 360), "#f5f5f5")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 46)
    heights = []
    widths = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    gap = 18
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = (360 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text(((640 - w) / 2, y), line, fill="#111111", font=font)
        y += h + gap
    img.save(path)


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    records = []
    for c in CASES:
        p = OUT_IMG / f"{c['id']}.png"
        render(c["lines"], p)
        records.append(
            {"id": c["id"], "text": c["text"], "label": c["label"], "image": str(p), "source": "synthetic-multimodal"}
        )
    OUT_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} cases -> {OUT_JSONL}; images -> {OUT_IMG}/")


if __name__ == "__main__":
    main()
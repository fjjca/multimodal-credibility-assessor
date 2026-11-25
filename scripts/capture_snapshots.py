import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent

CASES = [
    {
        "case_id": "mp04",
        "text": "这张照片拍摄于2024年上海马拉松比赛的现场。",
        "image": ROOT / "data" / "demo_photos" / "beijing_marathon.jpg",
        "snap": ROOT / "static" / "sample_result.json",
        "note": "图文矛盾 → 判定虚假：图中拱门写着 2024 北京半程马拉松，帖子却说是 2024 年上海马拉松",
    },
    {
        "case_id": "mp02",
        "text": "这张照片拍摄于巴黎埃菲尔铁塔下。",
        "image": ROOT / "data" / "demo_photos" / "paris_eiffel.jpg",
        "snap": ROOT / "static" / "sample_result_supported.json",
        "note": "图文一致 → 判定可信：图中就是巴黎埃菲尔铁塔，检索也证实铁塔位于巴黎",
    },
]


def run(case):
    with open(case["image"], "rb") as f:
        r = requests.post(
            f"{BASE}/api/verify",
            data={"text": case["text"]},
            files={"image": (case["image"].name, f, "image/jpeg")},
            timeout=60,
        )
    r.raise_for_status()
    task_id = r.json()["task_id"]
    deadline = time.time() + 1200
    while time.time() < deadline:
        time.sleep(5)
        s = requests.get(f"{BASE}/api/verify/{task_id}", timeout=30).json()
        if s["status"] in ("done", "failed", "error"):
            return s
    raise TimeoutError(case["case_id"])


for case in CASES:
    st = run(case)
    status = st["status"]
    print(case["case_id"], status, flush=True)
    if status != "done" or not st.get("result"):
        print("  FAILED:", str(st)[:400], flush=True)
        continue
    payload = dict(st["result"])
    payload["_meta"] = {
        "case_id": case["case_id"],
        "input_text": case["text"],
        "input_image": f"demo_photos/{case['image'].name}",
        "note": case["note"],
        "captured_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "disclaimer": "离线快照：一次真实跑批的结果，已固化到本地，断网也能演示；不是本次实时跑批。",
    }
    case["snap"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    g = payload.get("graph_elements") or {}
    sys.stdout.buffer.write(
        (
            "  -> "
            + str(payload.get("verdict"))
            + " "
            + str(payload.get("confidence"))
            + " | nodes="
            + str(len(g.get("nodes", [])))
            + " edges="
            + str(len(g.get("edges", [])))
            + " | "
            + str(payload.get("reason"))[:160]
            + "\n"
        ).encode("utf-8")
    )
    sys.stdout.flush()

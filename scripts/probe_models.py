"""模型配额体检：逐个探测当前配置的模型是否可用，定位 429 到底出在哪个模型上。

用法：
    .venv/Scripts/python -m scripts.probe_models
    .venv/Scripts/python -m scripts.probe_models --out out/model_probe.json

每个模型只发一次极小请求（不重试、不走缓存），故能在几十秒内给出全量体检结论。
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import APIStatusError, RateLimitError

from app.config import PROVIDERS, ROLES, get_settings
from app.llm.client import LLMClient

# judge_strong 以 model_override 方式调用，不在 ROLES 里，但同样是会 429 的升级目标
EXTRA_ROLES = ("judge_strong",)


def candidates(settings) -> list[tuple[str, str, str]]:
    """(标签角色, 客户端基础角色, provider:model) 列表：各角色主模型 + 备胎链，去重。

    judge_strong 不是 ROLES 成员，客户端按 judge 的基座发起调用（model_override 指定具体模型）。
    """
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    prim: list[tuple[str, str, str]] = []
    for role in ROLES:
        provider, model = settings.role_model(role)
        prim.append((role, role, f"{provider}:{model}"))
    for role in EXTRA_ROLES:
        cand = (getattr(settings, f"model_{role}", "") or "").strip()
        provider, _, model = cand.partition(":")
        if provider in PROVIDERS and model:
            prim.append((role, "judge", cand))

    for label, base_role, cand in prim:
        chain = [cand]
        if label in ROLES:
            chain += settings.role_chain(label)[1:]
        for c in chain:
            if c in seen:
                continue
            seen.add(c)
            out.append((label, base_role, c))
    return out


async def probe(client: LLMClient, role: str, base_role: str, candidate: str) -> dict:
    started = time.perf_counter()
    provider = candidate.partition(":")[0]
    extra_body = {"thinking": {"type": "disabled"}} if provider == "zhipu" else None
    try:
        r = await client.chat(
            base_role,
            [{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=8,
            extra_body=extra_body,
            model_override=candidate,
        )
        status, detail = "ok", (r.content or "")[:40]
    except RateLimitError:
        status, detail = "rate_limited", "429 配额/频率受限"
    except APIStatusError as exc:
        status, detail = f"error_{exc.status_code}", str(exc)[:120]
    except Exception as exc:  # 网络/配置类错误同样要可见
        status, detail = "error", f"{type(exc).__name__}: {exc}"[:120]
    return {
        "role": role,
        "model": candidate,
        "status": status,
        "detail": detail,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


async def main(out_path: str | None) -> int:
    # Windows 控制台默认 GBK，重定向到管道时中文会花屏；统一按 UTF-8 输出
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    probe_settings = settings.model_copy(
        update={
            "llm_role_fallbacks": {},
            "llm_retry_budgets": {},
            "llm_max_retries": 0,
            "llm_cache_mode": "off",
            "llm_min_interval_seconds": 0.5,
        }
    )
    client = LLMClient(probe_settings)
    rows = []
    for label, base_role, cand in candidates(settings):
        row = await probe(client, label, base_role, cand)
        rows.append(row)
        flag = {"ok": "OK  ", "rate_limited": "429 "}.get(row["status"], "ERR ")
        print(f"[{flag}] {row['role']:<12} {row['model']:<45} {row['latency_ms']:>6}ms  {row['detail']}")

    ok = sum(r["status"] == "ok" for r in rows)
    limited = sum(r["status"] == "rate_limited" for r in rows)
    failed = len(rows) - ok - limited
    print(f"\n合计 {len(rows)}：可用 {ok} / 限流 {limited} / 失败 {failed}")
    if limited:
        print("提示：限流模型可加入该角色的 LLM_ROLE_FALLBACKS 作为备胎，或换个时段再试（免费档高峰排队严重）。")

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "results": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已写入 {p}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型配额体检")
    parser.add_argument("--out", default="out/model_probe.json", help="体检结果 JSON 路径")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.out)))

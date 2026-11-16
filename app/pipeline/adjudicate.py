import json
import re

import networkx as nx
from openai import BadRequestError

from app.eval.schema import parse_verdict
from app.llm.client import LLMClient, LLMResult
from app.pipeline.graph import claim_evidence_summary
from app.pipeline.models import Claim

PROSECUTOR_SYSTEM = (
    "你是控方核查员。基于给定断言与证据图（含支持/矛盾证据及来源权重），论证该内容为何虚假或不可信。"
    "重点引用矛盾证据与低可信来源，但不得编造图中不存在的证据。"
    "只输出 JSON：{\"argument\": \"一段论证\", \"key_evidence\": [\"证据id\"]}。证据不足以反驳时如实说明。"
)
DEFENDER_SYSTEM = (
    "你是辩方核查员。基于给定断言与证据图，论证该内容为何真实可信。"
    "重点引用支持证据与高可信来源，但不得编造图中不存在的证据。"
    "只输出 JSON：{\"argument\": \"一段论证\", \"key_evidence\": [\"证据id\"]}。证据不足以支持时如实说明。"
)
JUDGE_SYSTEM = (
    "你是裁判。综合断言、证据图（支持/矛盾证据及来源权重）以及控辩双方论证做出最终裁决。"
    "证据图是裁决的数据源：优先依据加权证据本身，而非双方的修辞。"
    "verdict 取 supported/refuted/misleading/unverifiable 之一；证据不足或支持/矛盾均衡时输出 unverifiable 弃权。"
    "只输出 JSON：{\"verdict\": \"...\", \"confidence\": 0到1的小数, \"reason\": \"一句话理由\", \"evidence_chain\": [\"证据id\"]}。"
)

_SNIPPET = 200


def _graph_context(g: nx.Graph) -> str:
    summary = claim_evidence_summary(g)
    if not any(s["support"] or s["contradict"] for s in summary):
        return "（证据图中没有任何支持或矛盾证据）"
    lines = []
    for s in summary:
        lines.append(f"断言 {s['claim_id']}：{s['text']}")
        for rel, label in (("support", "支持"), ("contradict", "矛盾")):
            for item in s[rel]:
                node = g.nodes[item["evidence_id"]]
                snippet = (node.get("content") or node.get("title", ""))[:_SNIPPET]
                lines.append(
                    f"  [{label}] {item['evidence_id']} 来源={item['source_tier']} "
                    f"权重={item['weight']:.2f} {item['title']}：{snippet}"
                )
    return "\n".join(lines)


def _parse_argument(content: str) -> str:
    match = re.search(r"\{.*\}", content or "", re.S)
    if match:
        try:
            obj = json.loads(match.group(0), strict=False)
            if isinstance(obj, dict) and obj.get("argument"):
                return str(obj["argument"])
        except json.JSONDecodeError:
            pass
    return (content or "").strip()


def parse_judge_output(content: str) -> dict:
    match = re.search(r"\{.*\}", content or "", re.S)
    obj = {}
    if match:
        try:
            obj = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}
    try:
        conf = float(obj.get("confidence")) if obj.get("confidence") is not None else None
    except (TypeError, ValueError):
        conf = None
    chain = obj.get("evidence_chain") or []
    chain = [str(x) for x in chain] if isinstance(chain, list) else []
    return {
        "verdict": parse_verdict(str(obj.get("verdict", ""))),
        "confidence": conf,
        "reason": str(obj.get("reason", "")),
        "evidence_chain": chain,
    }


async def _argue(
    client: LLMClient, role: str, system: str, base: str
) -> tuple[str, LLMResult]:
    try:
        result = await client.chat(
            role,
            [{"role": "system", "content": system}, {"role": "user", "content": base}],
            temperature=0.3,
            max_tokens=512,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except BadRequestError:
        return "（内容审核拒答，无法论证）", LLMResult(content="", model="")
    return _parse_argument(result.content), result


async def adjudicate(
    client: LLMClient,
    claims: list[Claim],
    g: nx.Graph,
    enable_debate: bool = True,
) -> tuple[dict, list[LLMResult]]:
    claims_text = "\n".join(f"{c.id}: {c.text}" for c in claims) or "（无断言）"
    base = f"断言：\n{claims_text}\n\n证据图：\n{_graph_context(g)}"
    results: list[LLMResult] = []
    pros_arg = def_arg = "（未进行辩论）"
    if enable_debate:
        pros_arg, r1 = await _argue(client, "prosecutor", PROSECUTOR_SYSTEM, base)
        def_arg, r2 = await _argue(client, "defender", DEFENDER_SYSTEM, base)
        results.extend([r1, r2])
    judge_input = f"{base}\n\n控方论证：{pros_arg}\n\n辩方论证：{def_arg}"
    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_input},
    ]

    async def _judge(model_override: str | None = None) -> LLMResult:
        try:
            return await client.chat(
                "judge",
                judge_messages,
                temperature=0.0,
                max_tokens=512,
                extra_body={"thinking": {"type": "disabled"}},
                model_override=model_override,
            )
        except BadRequestError:
            return LLMResult(
                content='{"verdict":"unverifiable","confidence":null,"reason":"内容审核拒答，无法裁决"}',
                model="",
            )

    r3 = await _judge()
    results.append(r3)
    verdict = parse_judge_output(r3.content)
    verdict["escalated"] = False
    verdict["judge_model"] = r3.model

    settings = client.settings
    strong_model = settings.model_judge_strong.partition(":")[2]
    _, current_model = settings.role_model("judge")
    conf = verdict["confidence"]
    if (
        settings.enable_escalation
        and r3.model  # 内容审核降级(model="")不升级：强模型对同一内容同样拒答
        and strong_model != current_model
        and (conf is None or conf < settings.judge_confidence_threshold)
    ):
        r4 = await _judge(model_override=settings.model_judge_strong)
        results.append(r4)
        verdict = parse_judge_output(r4.content)
        verdict["escalated"] = True
        verdict["judge_model"] = r4.model or strong_model

    verdict["prosecutor"] = pros_arg
    verdict["defender"] = def_arg
    return verdict, results
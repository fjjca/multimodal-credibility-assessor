import json
import re

import networkx as nx
from openai import BadRequestError

from app.llm.client import LLMClient, LLMResult
from app.pipeline.graph import build_graph
from app.pipeline.models import RELATIONS, Claim, Evidence, Relation

RELATION_SYSTEM = (
    "你是证据关系判定器。给定一条断言和若干候选证据，逐条判断证据与断言的关系："
    "support=证据支持断言为真；contradict=证据反驳断言为假；neutral=不相关或无法判断。"
    "只输出一个 JSON 数组，元素形如 {\"evidence_id\": \"e1\", \"relation\": \"support\"}。"
    "必须为每条给出的证据输出一项，不要输出数组以外的内容。"
)

_SNIPPET = 300


def parse_relations(content: str, claim_id: str, valid_ids: set[str]) -> list[Relation]:
    match = re.search(r"\[.*\]", content or "", re.S)
    items = []
    if match:
        try:
            items = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            items = []
    out: list[Relation] = []
    seen: set[str] = set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        eid = str(it.get("evidence_id", ""))
        rel = str(it.get("relation", "")).strip().lower()
        if eid in valid_ids and eid not in seen and rel in RELATIONS:
            seen.add(eid)
            out.append(Relation(claim_id=claim_id, evidence_id=eid, relation=rel))
    return out


async def classify_relations(
    client: LLMClient, claim: Claim, evidences: list[Evidence]
) -> tuple[list[Relation], LLMResult | None]:
    if not evidences:
        return [], None
    lines = [f"断言（{claim.id}）：{claim.text}", "候选证据："]
    for e in evidences:
        snippet = (e.content or e.title)[:_SNIPPET]
        lines.append(f"- {e.id} [{e.source_tier}] {e.title}：{snippet}")
    messages = [
        {"role": "system", "content": RELATION_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
    try:
        result = await client.chat(
            "decompose",
            messages,
            temperature=0.0,
            max_tokens=512,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except BadRequestError:
        # 网页证据可能含敏感内容触发审核拒答：该断言不建边，流水线继续
        return [], None
    rels = parse_relations(result.content, claim.id, {e.id for e in evidences})
    return rels, result


async def build_evidence_graph(
    client: LLMClient,
    claims: list[Claim],
    evidences: list[Evidence],
    claim_map: dict[str, list[str]],
    *,
    image_evidence: Evidence | None = None,
    image_relations: list[Relation] | None = None,
) -> tuple[nx.Graph, list[Relation], list[LLMResult]]:
    by_id = {e.id: e for e in evidences}
    relations: list[Relation] = []
    results: list[LLMResult] = []
    for c in claims:
        evs = [by_id[i] for i in claim_map.get(c.id, []) if i in by_id]
        rels, res = await classify_relations(client, c, evs)
        relations.extend(rels)
        if res is not None:
            results.append(res)
    all_evidences = list(evidences)
    if image_evidence is not None:
        all_evidences.append(image_evidence)
    if image_relations:
        relations.extend(image_relations)
    g = build_graph(claims, all_evidences, relations)
    return g, relations, results
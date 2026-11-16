import json
from itertools import combinations
from pathlib import Path

import networkx as nx

from app.pipeline.models import (
    TIER_WEIGHTS,
    Claim,
    Evidence,
    Relation,
    domain_of,
)


def build_graph(
    claims: list[Claim],
    evidences: list[Evidence],
    relations: list[Relation],
) -> nx.Graph:
    """证据图：断言↔证据连支持/矛盾边(来源分级加权)，证据↔证据连同源边。

    图是裁决的数据源：裁决器读取的是这里的加权关系，而非原始检索列表。
    """
    g = nx.Graph()
    for c in claims:
        g.add_node(c.id, kind="claim", text=c.text)
    by_id = {e.id: e for e in evidences}
    for e in evidences:
        g.add_node(
            e.id,
            kind="evidence",
            title=e.title,
            url=e.url,
            content=e.content,
            source_tier=e.source_tier,
            score=e.score,
        )
    for r in relations:
        if r.relation not in ("support", "contradict"):
            continue
        if r.claim_id not in g or r.evidence_id not in by_id:
            continue
        weight = TIER_WEIGHTS.get(by_id[r.evidence_id].source_tier, 0.5)
        g.add_edge(r.claim_id, r.evidence_id, relation=r.relation, weight=weight)

    by_domain: dict[str, list[str]] = {}
    for e in evidences:
        dom = domain_of(e.url)
        if dom:
            by_domain.setdefault(dom, []).append(e.id)
    for ids in by_domain.values():
        for a, b in combinations(ids, 2):
            if not g.has_edge(a, b):
                g.add_edge(a, b, relation="same_source", weight=0.5)
    return g


def graph_stats(g: nx.Graph) -> dict:
    rels = [d.get("relation") for _, _, d in g.edges(data=True)]
    return {
        "n_claims": sum(d.get("kind") == "claim" for _, d in g.nodes(data=True)),
        "n_evidence": sum(d.get("kind") == "evidence" for _, d in g.nodes(data=True)),
        "n_support": rels.count("support"),
        "n_contradict": rels.count("contradict"),
        "n_same_source": rels.count("same_source"),
    }


def claim_evidence_summary(g: nx.Graph) -> list[dict]:
    """按断言聚合支持/矛盾证据(带权重)，供裁决器与前端证据链使用。"""
    out = []
    for node, data in g.nodes(data=True):
        if data.get("kind") != "claim":
            continue
        support, contradict = [], []
        for nb in g.neighbors(node):
            edge = g.edges[node, nb]
            nd = g.nodes[nb]
            if nd.get("kind") != "evidence":
                continue
            item = {
                "evidence_id": nb,
                "title": nd.get("title", ""),
                "url": nd.get("url", ""),
                "source_tier": nd.get("source_tier", "unknown"),
                "weight": edge.get("weight", 0.0),
            }
            if edge.get("relation") == "support":
                support.append(item)
            elif edge.get("relation") == "contradict":
                contradict.append(item)
        out.append(
            {"claim_id": node, "text": data.get("text", ""), "support": support, "contradict": contradict}
        )
    return out


def save_graph(g: nx.Graph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(nx.node_link_data(g, edges="edges"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_graph(path: str | Path) -> nx.Graph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges")


def graph_elements(g: nx.Graph) -> dict:
    """Cytoscape.js 元素：claim/evidence 节点 + support/contradict/same_source 边。"""
    nodes = []
    for nid, d in g.nodes(data=True):
        kind = d.get("kind", "evidence")
        label = d.get("text") if kind == "claim" else d.get("title", "")
        nodes.append(
            {
                "data": {
                    "id": nid,
                    "kind": kind,
                    "label": label or nid,
                    "source_tier": d.get("source_tier", ""),
                }
            }
        )
    edges = [
        {
            "data": {
                "id": f"e{i}",
                "source": a,
                "target": b,
                "relation": d.get("relation", ""),
                "weight": d.get("weight", 0.0),
            }
        }
        for i, (a, b, d) in enumerate(g.edges(data=True))
    ]
    return {"nodes": nodes, "edges": edges}
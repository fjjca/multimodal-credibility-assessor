import pytest

from app.pipeline.graph import (
    build_graph,
    claim_evidence_summary,
    graph_elements,
    graph_stats,
    load_graph,
    save_graph,
)
from app.pipeline.models import Claim, Evidence, Relation, classify_source_tier


@pytest.mark.parametrize(
    "url,tier",
    [
        ("https://www.gov.cn/zhengce/x.htm", "official"),
        ("https://who.int/news", "official"),
        ("https://www.bbc.com/news", "mainstream"),
        ("https://reuters.com/world", "mainstream"),
        ("https://www.reddit.com/r/x", "social"),
        ("https://weibo.com/u/1", "social"),
        ("https://example.com/post", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_source_tier(url, tier):
    assert classify_source_tier(url) == tier


def _fixture():
    claims = [Claim(id="c1", text="断言一"), Claim(id="c2", text="断言二")]
    evidences = [
        Evidence(id="e1", title="官方", url="https://www.gov.cn/a", content="x", source_tier="official"),
        Evidence(id="e2", title="社交", url="https://reddit.com/b", content="y", source_tier="social"),
        Evidence(id="e3", title="同域", url="https://reddit.com/c", content="z", source_tier="social"),
    ]
    relations = [
        Relation(claim_id="c1", evidence_id="e1", relation="support"),
        Relation(claim_id="c1", evidence_id="e2", relation="contradict"),
        Relation(claim_id="c2", evidence_id="e3", relation="neutral"),
    ]
    return claims, evidences, relations


def test_build_graph_edges_and_weights():
    g = build_graph(*_fixture())
    assert g.nodes["c1"]["kind"] == "claim"
    assert g.nodes["e1"]["kind"] == "evidence"
    # 官方支持边权重 1.0，社交矛盾边权重 0.4
    assert g.edges["c1", "e1"]["relation"] == "support"
    assert g.edges["c1", "e1"]["weight"] == pytest.approx(1.0)
    assert g.edges["c1", "e2"]["relation"] == "contradict"
    assert g.edges["c1", "e2"]["weight"] == pytest.approx(0.4)
    # neutral 建零权重边，仅供可视化，不进裁决汇总
    assert g.has_edge("c2", "e3")
    assert g.edges["c2", "e3"]["relation"] == "neutral"
    assert g.edges["c2", "e3"]["weight"] == 0.0
    # 同域证据 e2/e3 同源边
    assert g.edges["e2", "e3"]["relation"] == "same_source"


def test_graph_stats():
    g = build_graph(*_fixture())
    s = graph_stats(g)
    assert s["n_claims"] == 2
    assert s["n_evidence"] == 3
    assert s["n_support"] == 1
    assert s["n_contradict"] == 1
    assert s["n_same_source"] == 1
    assert s["n_neutral"] == 1


def test_neutral_edges_do_not_reach_the_judge():
    """中性边只用于画图：裁决用的支持/矛盾汇总里不应出现。"""
    g = build_graph(*_fixture())
    summary = {s["claim_id"]: s for s in claim_evidence_summary(g)}
    assert summary["c2"]["support"] == []
    assert summary["c2"]["contradict"] == []


def test_claim_evidence_summary():
    g = build_graph(*_fixture())
    summary = {s["claim_id"]: s for s in claim_evidence_summary(g)}
    assert len(summary["c1"]["support"]) == 1
    assert summary["c1"]["support"][0]["evidence_id"] == "e1"
    assert len(summary["c1"]["contradict"]) == 1
    assert summary["c2"]["support"] == []


def test_graph_json_roundtrip(tmp_path):
    g = build_graph(*_fixture())
    p = tmp_path / "graph.json"
    save_graph(g, p)
    g2 = load_graph(p)
    assert g2.nodes["e1"]["source_tier"] == "official"
    assert g2.edges["c1", "e1"]["weight"] == pytest.approx(1.0)
    assert graph_stats(g2) == graph_stats(g)


def test_graph_elements_cytoscape_shape():
    g = build_graph(*_fixture())
    el = graph_elements(g)
    nodes = {n["data"]["id"]: n["data"] for n in el["nodes"]}
    assert nodes["c1"]["kind"] == "claim" and nodes["c1"]["label"] == "断言一"
    assert nodes["e1"]["kind"] == "evidence" and nodes["e1"]["source_tier"] == "official"
    assert nodes["e1"]["url"] == "https://www.gov.cn/a" and nodes["e1"]["content"] == "x"
    assert "score" in nodes["e1"] and "url" not in nodes["c1"]
    rels = {(e["data"]["source"], e["data"]["target"]): e["data"] for e in el["edges"]}
    assert rels[("c1", "e1")]["relation"] == "support"
    assert rels[("c1", "e2")]["relation"] == "contradict"
    assert rels[("e2", "e3")]["relation"] == "same_source"
    assert all(e["data"]["id"] for e in el["edges"])
    assert len(el["nodes"]) == 5 and len(el["edges"]) == 4
    assert rels[("c2", "e3")]["relation"] == "neutral"
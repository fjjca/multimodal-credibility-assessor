import pytest

from app.pipeline.adjudicate import attach_graph_reasoning, _graph_context
from app.pipeline.graph import build_graph, claim_evidence_summary, graph_reasoning
from app.pipeline.graph_reasoning import (
    aggregate_graph_prior,
    apply_evidence_independence,
    propagate_evidence_belief,
)
from app.pipeline.models import Claim, Evidence, Relation


def _graph(claims, evidences, relations):
    return build_graph(claims, evidences, relations)


def test_same_source_evidence_is_discounted_per_claim():
    claims = [Claim(id="c1", text="x")]
    evidences = [
        Evidence(id="e1", title="转载1", url="https://news.example/a", source_tier="mainstream"),
        Evidence(id="e2", title="转载2", url="https://news.example/b", source_tier="mainstream"),
        Evidence(id="e3", title="独立来源", url="https://other.example/c", source_tier="mainstream"),
    ]
    relations = [Relation(claim_id="c1", evidence_id=e.id, relation="support") for e in evidences]
    g = _graph(claims, evidences, relations)
    stats = apply_evidence_independence(g)
    assert stats["n_discounted_edges"] == 2
    assert g.edges["c1", "e1"]["effective_weight"] == pytest.approx(0.8 / 2**0.5)
    assert g.edges["c1", "e2"]["effective_weight"] == pytest.approx(0.8 / 2**0.5)
    assert g.edges["c1", "e3"]["effective_weight"] == pytest.approx(0.8)
    assert g.edges["c1", "e1"]["weight"] == pytest.approx(0.8)


def test_same_source_discount_does_not_cross_claims():
    claims = [Claim(id="c1", text="x"), Claim(id="c2", text="y")]
    evidences = [
        Evidence(id="e1", title="a", url="https://news.example/a", source_tier="mainstream"),
        Evidence(id="e2", title="b", url="https://news.example/b", source_tier="mainstream"),
    ]
    relations = [
        Relation(claim_id="c1", evidence_id="e1", relation="support"),
        Relation(claim_id="c2", evidence_id="e2", relation="support"),
    ]
    g = _graph(claims, evidences, relations)
    apply_evidence_independence(g)
    assert g.edges["c1", "e1"]["effective_weight"] == pytest.approx(0.8)
    assert g.edges["c2", "e2"]["effective_weight"] == pytest.approx(0.8)


def test_summary_exposes_structured_metrics():
    claims = [Claim(id="c1", text="x")]
    evidences = [
        Evidence(id="e1", title="支持", url="https://gov.cn/a", source_tier="official"),
        Evidence(id="e2", title="矛盾", url="https://news.example/b", source_tier="mainstream"),
        Evidence(id="e3", title="无关", url="https://reddit.com/c", source_tier="social"),
    ]
    relations = [
        Relation(claim_id="c1", evidence_id="e1", relation="support"),
        Relation(claim_id="c1", evidence_id="e2", relation="contradict"),
        Relation(claim_id="c1", evidence_id="e3", relation="neutral"),
    ]
    g = _graph(claims, evidences, relations)
    summary = claim_evidence_summary(g)[0]
    assert summary["support_weight"] == pytest.approx(1.0)
    assert summary["contradict_weight"] == pytest.approx(0.8)
    assert summary["net_weight"] == pytest.approx(0.2)
    assert summary["evidence_coverage"] == pytest.approx(2 / 3, abs=5e-5)
    assert summary["graph_prior"] > 0
    assert summary["support"][0]["independence"] == 1.0


def test_propagation_couples_claims_through_shared_evidence():
    claims = [Claim(id="c1", text="x"), Claim(id="c2", text="y")]
    evidences = [
        Evidence(id="e1", title="反证", url="https://gov.cn/a", source_tier="official"),
        Evidence(id="e2", title="共享证据", url="https://who.int/b", source_tier="official"),
    ]
    relations = [
        Relation(claim_id="c1", evidence_id="e1", relation="contradict"),
        Relation(claim_id="c1", evidence_id="e2", relation="support"),
        Relation(claim_id="c2", evidence_id="e2", relation="support"),
    ]
    g = _graph(claims, evidences, relations)
    reasoning = propagate_evidence_belief(g)
    assert reasoning["converged"] is True
    assert reasoning["claims"]["c2"]["graph_prior"] > 0
    assert aggregate_graph_prior(reasoning)["evidence_coverage"] == 1.0


def test_empty_graph_has_no_prior():
    g = _graph([Claim(id="c1", text="x")], [], [])
    reasoning = propagate_evidence_belief(g)
    assert reasoning["claims"]["c1"]["prior_label"] == "insufficient"
    assert aggregate_graph_prior(reasoning)["graph_prior"] is None


def test_graph_context_contains_structured_metrics():
    claims = [Claim(id="c1", text="x")]
    evidences = [Evidence(id="e1", title="反证", url="https://gov.cn/a", source_tier="official")]
    g = _graph(claims, evidences, [Relation(claim_id="c1", evidence_id="e1", relation="contradict")])
    context = _graph_context(g)
    assert "结构化图指标" in context
    assert "去偏支持权重" in context
    assert "图先验" in context


def test_graph_conflict_penalizes_but_does_not_override_verdict():
    claims = [Claim(id="c1", text="x")]
    evidences = [Evidence(id="e1", title="反证", url="https://gov.cn/a", source_tier="official")]
    g = _graph(claims, evidences, [Relation(claim_id="c1", evidence_id="e1", relation="contradict")])
    verdict = {"verdict": "supported", "confidence": 0.9}
    attach_graph_reasoning(verdict, g, conflict_penalty=0.2)
    assert verdict["verdict"] == "supported"
    assert verdict["confidence_raw"] == pytest.approx(0.9)
    assert verdict["confidence"] == pytest.approx(0.72)
    assert verdict["graph_disagreement"] is True


def test_graph_reasoning_can_be_disabled_for_ablation():
    claims = [Claim(id="c1", text="x")]
    evidences = [Evidence(id="e1", title="支持", url="https://gov.cn/a", source_tier="official")]
    g = _graph(claims, evidences, [Relation(claim_id="c1", evidence_id="e1", relation="support")])
    result = graph_reasoning(g, independence=False)
    assert result["independence"]["enabled"] is False
    assert result["aggregate"]["prior_label"] == "support"

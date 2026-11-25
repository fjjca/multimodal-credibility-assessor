"""证据图推理：同源去偏、断言聚合与二分图可信度传播。"""

import math

import networkx as nx

_ACTIVE = {"support", "contradict"}
_SIGN = {"support": 1.0, "contradict": -1.0}


def apply_evidence_independence(g: nx.Graph) -> dict:
    """同一来源簇内的多条有效证据按 1/sqrt(n) 折扣，避免转载重复计票。"""
    active = [(a, b, d) for a, b, d in g.edges(data=True) if d.get("relation") in _ACTIVE]
    clusters: dict[str, int] = {}
    for a, b, _ in active:
        for node in (a, b):
            if g.nodes[node].get("kind") == "evidence" and node not in clusters:
                clusters[node] = node
    for a, b, d in g.edges(data=True):
        if d.get("relation") != "same_source":
            continue
        ra, rb = clusters.get(a), clusters.get(b)
        if ra is not None and rb is not None and ra != rb:
            lo, hi = sorted((ra, rb))
            for node, root in list(clusters.items()):
                if root == hi:
                    clusters[node] = lo

    by_claim_cluster: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for a, b, d in active:
        claim, evidence = (a, b) if g.nodes[a].get("kind") == "claim" else (b, a)
        by_claim_cluster.setdefault((claim, clusters[evidence]), []).append((evidence, d))

    discounted = 0
    max_cluster = 1
    for (claim, cluster), items in by_claim_cluster.items():
        n = len(items)
        max_cluster = max(max_cluster, n)
        factor = 1.0 / math.sqrt(n) if n > 1 else 1.0
        if factor < 1.0:
            discounted += len(items)
        for evidence, data in items:
            raw = float(data.get("weight", 0.0))
            data["raw_weight"] = raw
            data["effective_weight"] = raw * factor
            data["independence"] = factor
            data["source_cluster"] = cluster
            data["independence_claim"] = claim
    return {
        "enabled": True,
        "n_discounted_edges": discounted,
        "max_active_cluster": max_cluster,
    }


def _tanh(x: float) -> float:
    return math.tanh(max(-10.0, min(10.0, x)))


def propagate_evidence_belief(
    g: nx.Graph,
    *,
    damping: float = 0.85,
    max_iter: int = 200,
    tolerance: float = 1e-6,
) -> dict:
    """在断言↔证据二分图上做阻尼符号传播，返回每条断言的图先验。"""
    claims = [n for n, d in g.nodes(data=True) if d.get("kind") == "claim"]
    evidence = [n for n, d in g.nodes(data=True) if d.get("kind") == "evidence"]
    initial: dict[str, float] = {c: 0.0 for c in claims}
    incident: dict[str, list[tuple[str, float, float]]] = {n: [] for n in claims + evidence}
    active_neighbors = {c: 0 for c in claims}
    for a, b, d in g.edges(data=True):
        rel = d.get("relation")
        if rel not in _ACTIVE:
            continue
        claim, ev = (a, b) if g.nodes[a].get("kind") == "claim" else (b, a)
        weight = float(d.get("effective_weight", d.get("weight", 0.0)))
        sign = _SIGN[rel]
        incident[claim].append((ev, weight, sign))
        incident[ev].append((claim, weight, sign))
        active_neighbors[claim] += 1
        initial[claim] += sign * weight

    for c in claims:
        initial[c] = _tanh(initial[c] / 2.0)
    current = {**initial, **{e: 0.0 for e in evidence}}
    converged = not any(incident.values())
    iterations = 0
    for i in range(max_iter):
        iterations = i + 1
        updated = dict(current)
        for node in evidence:
            neighbors = incident[node]
            total = sum(w for _, w, _ in neighbors)
            target = sum(w * sign * current[nb] for nb, w, sign in neighbors) / total if total else current[node]
            updated[node] = damping * current[node] + (1.0 - damping) * target
        for node in claims:
            neighbors = incident[node]
            total = sum(w for _, w, _ in neighbors)
            target = sum(w * sign * updated[nb] for nb, w, sign in neighbors) / total if total else current[node]
            updated[node] = damping * current[node] + (1.0 - damping) * target
        delta = max((abs(updated[n] - current[n]) for n in current), default=0.0)
        current = updated
        if delta < tolerance:
            converged = True
            break

    priors: dict[str, dict] = {}
    for c in claims:
        total_evidence = sum(1 for nb in g.neighbors(c) if g.nodes[nb].get("kind") == "evidence")
        active = active_neighbors[c]
        coverage = active / total_evidence if total_evidence else 0.0
        prior = _tanh(current[c] * 1.5)
        label = "insufficient" if not active else "support" if prior > 0.15 else "contradict" if prior < -0.15 else "balanced"
        priors[c] = {
            "graph_prior": round(prior, 4),
            "prior_label": label,
            "evidence_coverage": round(coverage, 4),
            "active_evidence": active,
            "candidate_evidence": total_evidence,
        }
    return {
        "claims": priors,
        "iterations": iterations,
        "converged": converged,
        "damping": damping,
    }


def aggregate_graph_prior(reasoning: dict) -> dict:
    claims = reasoning.get("claims", {})
    values = [v["graph_prior"] for v in claims.values() if v["active_evidence"]]
    if not values:
        return {"graph_prior": None, "prior_label": "insufficient", "evidence_coverage": 0.0}
    prior = sum(values) / len(values)
    coverage = sum(v["evidence_coverage"] for v in claims.values()) / max(len(claims), 1)
    label = "support" if prior > 0.15 else "contradict" if prior < -0.15 else "balanced"
    return {
        "graph_prior": round(prior, 4),
        "prior_label": label,
        "evidence_coverage": round(coverage, 4),
    }

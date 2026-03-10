"""
Information-retrieval evaluation metrics.

All functions accept:
  retrieved  – ordered list of doc_ids returned by the retriever
  relevant   – set (or list) of doc_ids known to be relevant for the query

Metrics implemented:
  precision_at_k      – fraction of top-k retrieved that are relevant
  recall_at_k         – fraction of all relevant docs that appear in top-k
  mrr                 – Mean Reciprocal Rank (rank of first relevant result)
  ndcg_at_k           – normalised Discounted Cumulative Gain
  average_precision   – area-under-precision-recall curve proxy (AP)
  distance_score      – cosine distance between query and retrieved passage
                        (lower = more similar; high value flags hallucination)
  credibility_score   – heuristic source-reliability score 0-100
                        (higher = more credible; low value flags misinformation)
"""

from __future__ import annotations

import math
import re


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """P@k: how many of the top-k returned docs are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """R@k: fraction of relevant docs captured in the top-k results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank for a single query.

    Returns 1/rank of the first relevant doc, or 0 if none found.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """nDCG@k.

    Binary relevance: rel=1 if doc in relevant set, else 0.
    DCG  = sum_i ( rel_i / log2(i+1) )  for i in 1..k
    IDCG = sum_i ( 1   / log2(i+1) )    for i in 1..min(|relevant|, k)
    """
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision (AP) for a single query – area under P/R curve."""
    if not relevant:
        return 0.0
    hits = 0
    running_precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            running_precision_sum += hits / rank
    return running_precision_sum / len(relevant)


# ---------------------------------------------------------------------------
# Distance Score
# ---------------------------------------------------------------------------

def distance_score(
    query_vec: "np.ndarray",
    passage_vec: "np.ndarray",
) -> float:
    """Cosine distance between a query and a retrieved passage embedding.

    distance = 1 - cosine_similarity ∈ [0, 2]

    Interpretation
    --------------
    0.0  – perfect semantic match (identical vectors)
    1.0  – orthogonal (no semantic overlap)
    2.0  – opposite meaning

    Hallucination flag: if distance > HALLUCINATION_DISTANCE_THRESHOLD, the
    passage is semantically too far from the query to ground a valid answer.
    """
    import numpy as np
    q = query_vec.flatten().astype(float)
    p = passage_vec.flatten().astype(float)
    q_norm = np.linalg.norm(q)
    p_norm = np.linalg.norm(p)
    if q_norm == 0 or p_norm == 0:
        return 1.0  # undefined → treat as orthogonal
    cos_sim = float(np.dot(q, p) / (q_norm * p_norm))
    return 1.0 - cos_sim


# Threshold above which we flag a retrieved passage as a potential hallucination source
HALLUCINATION_DISTANCE_THRESHOLD = 0.7


def hallucination_flagged(dist: float, threshold: float = HALLUCINATION_DISTANCE_THRESHOLD) -> bool:
    """Return True if the distance score exceeds the hallucination threshold."""
    return dist > threshold


# ---------------------------------------------------------------------------
# Credibility Score
# ---------------------------------------------------------------------------

# Heuristic keyword lists for credibility scoring
_HIGH_CRED_PATTERNS = [
    r"\bwikipedia\b", r"\bpubmed\b", r"\bnih\.gov\b", r"\bnature\.com\b",
    r"\bsciencedirect\b", r"\bjstor\b", r"\breuters\b", r"\bap news\b",
    r"\bassociated press\b", r"\bgovernment\b", r"\buniversity\b",
    r"\bjournal\b", r"\bstudy\b", r"\bresearch\b", r"\bstudy found\b",
    r"\baccording to researchers\b", r"\bpeer.reviewed\b",
]
_LOW_CRED_PATTERNS = [
    r"\bfake news\b", r"\bconspiracy\b", r"\bunverified\b", r"\brumour\b",
    r"\brunaway\b", r"\bclickbait\b", r"\bsensational\b", r"\bhoax\b",
    r"\bmisinformation\b", r"\bsatire\b",
]
# Bonus/penalty weights
_HIGH_CRED_BONUS = 8    # per matching high-credibility pattern
_LOW_CRED_PENALTY = 15  # per matching low-credibility pattern
_BASE_CREDIBILITY = 50  # neutral starting point


def credibility_score(passage_text: str) -> float:
    """Heuristic credibility score for a passage in [0, 100].

    Algorithm
    ---------
    Start at 50 (neutral).
    +8  for each high-credibility signal (research/academic language, known
        trusted sources, citation markers).
    -15 for each low-credibility signal (misinformation keywords, sensational
        language, known fake-news markers).
    Clamp to [0, 100].

    Interpretation
    --------------
    ≥ 80 : highly credible (trusted academic / news source language)
    50-79: neutral / unknown credibility
    < 50 : potentially unreliable — flag for review

    Note: This is a fast lexical heuristic suitable for a streaming pipeline.
    Production systems would use a pre-trained source-credibility classifier.
    """
    text_lower = passage_text.lower()
    score = float(_BASE_CREDIBILITY)

    for pat in _HIGH_CRED_PATTERNS:
        if re.search(pat, text_lower):
            score += _HIGH_CRED_BONUS

    for pat in _LOW_CRED_PATTERNS:
        if re.search(pat, text_lower):
            score -= _LOW_CRED_PENALTY

    return max(0.0, min(100.0, score))


# Threshold below which we flag a passage as potentially misinformative
MISINFORMATION_CREDIBILITY_THRESHOLD = 40.0


def misinformation_flagged(
    cred: float,
    threshold: float = MISINFORMATION_CREDIBILITY_THRESHOLD,
) -> bool:
    """Return True if credibility score is below the misinformation threshold."""
    return cred < threshold


# ---------------------------------------------------------------------------
# Unified metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    retrieved: list[str],
    relevant: set[str],
    k: int = 10,
) -> dict[str, float]:
    """Convenience wrapper – returns a dict with all IR metrics for one query."""
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
        "ap": average_precision(retrieved, relevant),
    }

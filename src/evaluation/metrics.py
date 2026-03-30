"""
Evaluation metrics for Arabic text chunking.

Primary metrics (for benchmark use)
-------------------------------------
precision_at_k          — RAG retrieval quality: is the answer in the top-K chunks?
chunk_coherence_score   — Avg concept confidence across all chunks (requires tagger).
concept_purity          — Fraction of consecutive sentence pairs that share a concept.

Supporting metrics
------------------
recall_at_k             — Fraction of relevant chunks found in top-K.
f1_at_k                 — Harmonic mean of P@K and R@K.
boundary_precision_recall_f1  — Boundary detection quality with position tolerance.
EvaluationReport        — Aggregates metrics over a corpus run.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_SENT_SPLIT = re.compile(r'[.!?؟،\n]+')


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Primary benchmark metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_chunks: List[Dict],
    relevant_chunk: str,
    k: int = 3,
) -> float:
    """
    RAG retrieval precision: does the answer appear in the top-K chunks?

    Parameters
    ----------
    retrieved_chunks:
        Ordered list of chunk dicts (highest similarity first).
    relevant_chunk:
        The answer text to search for (substring match).
    k:
        Number of top chunks to consider.

    Returns
    -------
    1.0 if ``relevant_chunk`` is a substring of any of the top-K chunk
    texts (case-insensitive), else 0.0.
    """
    needle = relevant_chunk.strip().lower()
    for chunk in retrieved_chunks[:k]:
        if needle in chunk.get("text", "").lower():
            return 1.0
    return 0.0


def chunk_coherence_score(
    chunks: List[Dict],
    concept_tagger,
) -> float:
    """
    Average concept-confidence across all chunks.

    A chunk is coherent if the tagger assigns it a confident, specific
    concept label.  Low-confidence or "عام/General" labels pull the
    score down.

    Parameters
    ----------
    chunks:
        List of chunk dicts (as returned by any chunker's chunk_dicts()).
    concept_tagger:
        A ConceptTagger instance used to re-tag each chunk.

    Returns
    -------
    Mean confidence in [0.0, 1.0], or NaN if ``chunks`` is empty.
    """
    if not chunks:
        return float("nan")

    scores: List[float] = []
    for c in chunks:
        text = c.get("text", "")
        if not text.strip():
            continue
        # Tag the whole chunk as a single-element "window"
        result = concept_tagger.tag([text])
        scores.append(result.confidence)

    return round(float(np.mean(scores)), 4) if scores else float("nan")


def concept_purity(
    chunks: List[Dict],
    concept_tagger,
) -> float:
    """
    Intra-chunk concept purity: fraction of consecutive sentence pairs
    that share the same concept label.

    A pure chunk has all its sentences discussing the same topic.
    Impure chunks mix topics, indicating the chunker failed to place a
    boundary at the right place.

    Parameters
    ----------
    chunks:
        List of chunk dicts.
    concept_tagger:
        A ConceptTagger instance used to tag individual sentences.

    Returns
    -------
    Float in [0.0, 1.0].  Returns 1.0 when all chunks have < 2 sentences
    (trivially pure), or NaN when chunks is empty.
    """
    if not chunks:
        return float("nan")

    total_pairs = 0
    matching_pairs = 0

    for c in chunks:
        sentences = _split_sentences(c.get("text", ""))
        if len(sentences) < 2:
            continue
        # Tag each sentence individually
        labels = [concept_tagger.tag([s]).concept for s in sentences]
        for i in range(len(labels) - 1):
            total_pairs += 1
            if labels[i] == labels[i + 1]:
                matching_pairs += 1

    if total_pairs == 0:
        return 1.0  # no pairs to compare → trivially pure
    return round(matching_pairs / total_pairs, 4)


# ---------------------------------------------------------------------------
# Supporting retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Fraction of the relevant set found within the top-K retrieved items."""
    if not relevant:
        return 1.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Harmonic mean of precision@K and recall@K."""
    p = sum(1 for item in retrieved[:k] if item in relevant) / max(k, 1)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Boundary detection metrics
# ---------------------------------------------------------------------------

def boundary_precision_recall_f1(
    pred_boundaries: List[int],
    gold_boundaries: List[int],
    n_sentences: int,
    window: int = 1,
) -> Tuple[float, float, float]:
    """
    Boundary-level precision, recall, and F1 with a position-tolerance window.

    A predicted boundary at position p is a True Positive if a gold
    boundary exists within [p - window, p + window].

    Returns
    -------
    (precision, recall, f1) — all in [0, 1].
    """
    gold_set = set(gold_boundaries)

    def near(pos: int) -> bool:
        return any(abs(pos - g) <= window for g in gold_set)

    tp = sum(1 for p in pred_boundaries if near(p))
    fp = len(pred_boundaries) - tp
    fn = sum(
        1 for g in gold_boundaries
        if not any(abs(p - g) <= window for p in pred_boundaries)
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """
    Aggregates evaluation metrics over an entire chunking run.

    Usage
    -----
    >>> report = EvaluationReport()
    >>> report.add(pred_boundaries=[3, 7], gold_boundaries=[3, 8], n_sentences=12)
    >>> print(report.summary())
    """

    _p1: List[float]        = field(default_factory=list, repr=False)
    _p3: List[float]        = field(default_factory=list, repr=False)
    _coherence: List[float] = field(default_factory=list, repr=False)
    _purity: List[float]    = field(default_factory=list, repr=False)
    _bp: List[float]        = field(default_factory=list, repr=False)
    _br: List[float]        = field(default_factory=list, repr=False)
    _bf: List[float]        = field(default_factory=list, repr=False)

    def add_retrieval(
        self,
        retrieved_chunks: List[Dict],
        answer: str,
    ) -> None:
        """Record P@1 and P@3 for one QA pair."""
        self._p1.append(precision_at_k(retrieved_chunks, answer, k=1))
        self._p3.append(precision_at_k(retrieved_chunks, answer, k=3))

    def add_boundary(
        self,
        pred_boundaries: List[int],
        gold_boundaries: List[int],
        n_sentences: int,
        window: int = 1,
    ) -> None:
        p, r, f = boundary_precision_recall_f1(
            pred_boundaries, gold_boundaries, n_sentences, window
        )
        self._bp.append(p)
        self._br.append(r)
        self._bf.append(f)

    def add_coherence(self, score: float) -> None:
        if not math.isnan(score):
            self._coherence.append(score)

    def add_purity(self, score: float) -> None:
        if not math.isnan(score):
            self._purity.append(score)

    def _mean(self, values: List[float]) -> float:
        return round(float(np.mean(values)), 4) if values else float("nan")

    def summary(self) -> Dict[str, float]:
        return {
            "precision_at_1":       self._mean(self._p1),
            "precision_at_3":       self._mean(self._p3),
            "chunk_coherence":      self._mean(self._coherence),
            "concept_purity":       self._mean(self._purity),
            "boundary_precision":   self._mean(self._bp),
            "boundary_recall":      self._mean(self._br),
            "boundary_f1":          self._mean(self._bf),
        }

    def __repr__(self) -> str:
        lines = ["EvaluationReport("]
        for k, v in self.summary().items():
            lines.append(f"  {k}: {v}")
        lines.append(")")
        return "\n".join(lines)

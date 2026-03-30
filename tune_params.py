"""
Parameter sweep for OntologyChunker.

Tries combinations of threshold, window_size, and mode-specific parameters,
scores each with unsupervised metrics, and prints a ranked table.

Usage
-----
    # Fast: keyword-only mode (no model downloads)
    python tune_params.py arabic.test --mode keyword

    # With ensemble models (slower, downloads models first run)
    python tune_params.py arabic.test --mode hybrid

    # Both
    python tune_params.py arabic.test --mode both

Metrics used (no ground truth needed)
--------------------------------------
coherence  — mean concept confidence across all chunks (higher = each chunk
             is clearly about one topic).
contrast   — fraction of adjacent chunk pairs with DIFFERENT concepts
             (higher = the chunker is actually splitting on topic changes).
balance    — 1 - normalised std of chunk sizes.  Penalises one giant chunk
             or many single-sentence fragments.
composite  — 0.40 × coherence + 0.40 × contrast + 0.20 × balance
             (the primary ranking score).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.chunker.ontology_chunker import OntologyChunker
from src.chunker.ensemble_embedder import EnsembleEmbedder
from src.evaluation.metrics import chunk_coherence_score
from src.chunker.concept_tagger import ConceptTagger


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _concept_contrast(chunks: List[Dict]) -> float:
    """Fraction of adjacent chunk pairs that have DIFFERENT concept labels."""
    if len(chunks) < 2:
        return 0.0
    pairs = len(chunks) - 1
    different = sum(
        1 for i in range(pairs)
        if chunks[i]["concept"] != chunks[i + 1]["concept"]
    )
    return round(different / pairs, 4)


def _size_balance(chunks: List[Dict]) -> float:
    """
    1 - coefficient_of_variation(chunk_sizes), clamped to [0, 1].
    High = chunks are evenly sized.  Low = one giant + many tiny fragments.
    """
    sizes = [c.get("sentence_count", 1) for c in chunks]
    if len(sizes) < 2:
        return 0.0
    mean = np.mean(sizes)
    if mean == 0:
        return 0.0
    cv = np.std(sizes) / mean
    return round(max(0.0, 1.0 - cv), 4)


def score(chunks: List[Dict], tagger: ConceptTagger) -> Dict[str, float]:
    n = len(chunks)
    if n == 0:
        return {"coherence": 0, "contrast": 0, "balance": 0, "composite": 0,
                "n_chunks": 0, "avg_sentences": 0}

    coherence = chunk_coherence_score(chunks, tagger)
    contrast  = _concept_contrast(chunks)
    balance   = _size_balance(chunks)
    composite = round(0.40 * coherence + 0.40 * contrast + 0.20 * balance, 4)
    avg_sents = round(np.mean([c.get("sentence_count", 1) for c in chunks]), 1)

    return {
        "coherence":    round(coherence, 3),
        "contrast":     round(contrast, 3),
        "balance":      round(balance, 3),
        "composite":    composite,
        "n_chunks":     n,
        "avg_sentences": avg_sents,
    }


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------

def _run(text: str, tagger: ConceptTagger, shared_embedder=None, **kwargs) -> Dict[str, float]:
    """Instantiate a chunker with given kwargs, chunk text, return scores."""
    try:
        chunker = OntologyChunker(
            min_sentences=2,
            max_sentences=30,
            embedder=shared_embedder,
            **kwargs,
        )
        chunks = chunker.chunk_dicts(text)
        return score(chunks, tagger)
    except Exception as exc:
        return {"error": str(exc), "composite": -1.0}


def keyword_sweep(text: str, tagger: ConceptTagger) -> List[Dict]:
    thresholds   = [0.5, 0.6, 0.7, 0.8, 0.9]
    window_sizes = [2, 3, 4, 5]
    conf_drops   = [0.20, 0.30, 0.40, 0.50]

    results = []
    total = len(thresholds) * len(window_sizes) * len(conf_drops)
    done = 0

    for t in thresholds:
        for w in window_sizes:
            for cd in conf_drops:
                s = _run(
                    text, tagger,
                    use_ensemble=False,
                    shift_threshold=t,
                    window_size=w,
                    confidence_drop_threshold=cd,
                )
                s.update({"mode": "keyword", "threshold": t,
                           "window": w, "conf_drop": cd})
                results.append(s)
                done += 1
                print(f"\r  keyword sweep {done}/{total} ...", end="", flush=True)

    print()
    return results


def hybrid_sweep(text: str, tagger: ConceptTagger) -> List[Dict]:
    thresholds   = [0.2, 0.3, 0.4, 0.5]
    window_sizes = [2, 3, 4]
    ont_weights  = [0.25, 0.5, 0.75]

    # Load all 3 models ONCE — shared across every combo in this sweep
    print("  Loading embedding models (once) ...")
    shared_embedder = EnsembleEmbedder()
    shared_embedder._load_all()  # force eager load so the progress bar shows here
    print("  Models ready.\n")

    results = []
    total = len(thresholds) * len(window_sizes) * len(ont_weights)
    done = 0

    for t in thresholds:
        for w in window_sizes:
            for ow in ont_weights:
                s = _run(
                    text, tagger,
                    shared_embedder=shared_embedder,
                    use_ensemble=True,
                    shift_threshold=t,
                    window_size=w,
                    ontology_weight=ow,
                )
                s.update({"mode": "hybrid", "threshold": t,
                           "window": w, "ont_weight": ow})
                results.append(s)
                done += 1
                print(f"\r  hybrid sweep {done}/{total} ...", end="", flush=True)

    print()
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_table(rows: List[Dict], top_n: int = 15) -> None:
    valid = [r for r in rows if r.get("composite", -1) >= 0]
    valid.sort(key=lambda r: r["composite"], reverse=True)
    valid = valid[:top_n]

    if not valid:
        print("  (no valid results)")
        return

    # Header
    col_w = {"rank": 4, "mode": 8, "thresh": 7, "window": 6,
              "param3": 10, "comp": 6, "coh": 6, "cont": 6,
              "bal": 6, "chunks": 6, "avg_s": 6}

    def fmt_row(rank, r):
        mode = r.get("mode", "?")
        param3 = (
            f"cd={r['conf_drop']}" if mode == "keyword"
            else f"ow={r.get('ont_weight', '?')}"
        )
        return (
            f"  {rank:<4} {mode:<8} {r['threshold']:<7.2f} {r['window']:<6} "
            f"{param3:<10} {r['composite']:<6.3f} {r['coherence']:<6.3f} "
            f"{r['contrast']:<6.3f} {r['balance']:<6.3f} "
            f"{r['n_chunks']:<6} {r['avg_sentences']:<6}"
        )

    header = (
        f"  {'#':<4} {'mode':<8} {'thresh':<7} {'win':<6} "
        f"{'param':<10} {'comp':<6} {'coh':<6} {'cont':<6} "
        f"{'bal':<6} {'chunks':<6} {'avg_s':<6}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)
    for i, r in enumerate(valid, 1):
        print(fmt_row(i, r))

    best = valid[0]
    print()
    print("  Best config:")
    print(f"    mode      = {best['mode']}")
    print(f"    threshold = {best['threshold']}")
    print(f"    window    = {best['window']}")
    if best["mode"] == "keyword":
        print(f"    conf_drop = {best['conf_drop']}")
    else:
        print(f"    ont_weight= {best.get('ont_weight')}")
    print(f"    composite = {best['composite']}  "
          f"(coherence={best['coherence']}, contrast={best['contrast']}, "
          f"balance={best['balance']})")
    print(f"    → {best['n_chunks']} chunks, avg {best['avg_sentences']} sentences each")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OntologyChunker parameter sweep")
    parser.add_argument("file", help="Arabic .txt file to chunk")
    parser.add_argument("--mode", choices=["keyword", "hybrid", "both"],
                        default="keyword",
                        help="Which mode(s) to sweep (default: keyword)")
    parser.add_argument("--top", type=int, default=15,
                        help="Number of top results to display (default: 15)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    tagger = ConceptTagger(use_morphology=False)

    all_results: List[Dict] = []

    if args.mode in ("keyword", "both"):
        print("Running keyword-mode sweep ...")
        all_results += keyword_sweep(text, tagger)

    if args.mode in ("hybrid", "both"):
        print("Running hybrid-mode sweep (loads 3 models on first run) ...")
        all_results += hybrid_sweep(text, tagger)

    print()
    print(f"=== Top {args.top} configurations ===")
    _print_table(all_results, top_n=args.top)


if __name__ == "__main__":
    main()

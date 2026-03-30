"""
Benchmark all chunkers head-to-head on one or more Arabic text files.

Usage
-----
    # Unsupervised (no ground truth needed — works on any file)
    python benchmark.py arabic.test
    python benchmark.py arabic.test test.txt

    # Supervised (you supply known boundary positions)
    python benchmark.py arabic.test --boundaries "6,11,15"
    # Boundary positions = sentence indices where a NEW chunk should start
    # e.g. "6,11,15" means new chunks begin at sentences 6, 11, and 15

    # Skip slow models (no sentence-transformers needed)
    python benchmark.py arabic.test --no-semantic --no-hybrid

Chunkers compared
-----------------
    ontology-kw    OntologyChunker  keyword mode   (no models)
    ontology-hyb   OntologyChunker  hybrid mode    (3 models)
    semantic       SemanticChunker  SBERT-100K     (1 model)
    fixed          FixedChunker     token windows  (no models)
    recursive      RecursiveChunker char windows   (no models)

Metrics
-------
    coherence    Mean concept confidence per chunk — needs ontology.
                 For non-ontology chunkers this reflects how clearly
                 each chunk maps to ONE domain when re-tagged.
    purity       Fraction of adjacent sentence pairs inside a chunk
                 that share the same concept label (intra-chunk topic
                 consistency).  1.0 = perfect.
    contrast     Fraction of adjacent chunk pairs with DIFFERENT concept
                 labels.  1.0 = every boundary separates a topic change.
    balance      1 - CV(chunk_sizes).  Penalises extreme size variation.
    composite    0.35·coherence + 0.35·purity + 0.20·contrast + 0.10·balance

    boundary_P/R/F1  (only when --boundaries provided)
                 How accurately the chunker finds the reference boundaries
                 (±1 sentence tolerance window).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.chunker.concept_tagger import ConceptTagger
from src.chunker.ontology_chunker import OntologyChunker
from src.baselines.fixed_chunker import FixedChunker
from src.baselines.recursive_chunker import RecursiveChunker
from src.evaluation.metrics import (
    chunk_coherence_score,
    concept_purity,
    boundary_precision_recall_f1,
)
from src.preprocessing.tokenizer import sentence_tokenize
from src.preprocessing.normalizer import normalize


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _concept_contrast(chunks: List[Dict]) -> float:
    if len(chunks) < 2:
        return 0.0
    pairs = len(chunks) - 1
    diff = sum(
        1 for i in range(pairs)
        if chunks[i].get("concept") != chunks[i + 1].get("concept")
    )
    return round(diff / pairs, 4)


def _size_balance(chunks: List[Dict]) -> float:
    sizes = [c.get("sentence_count", 1) for c in chunks]
    if len(sizes) < 2:
        return 0.0
    mean = np.mean(sizes)
    if mean == 0:
        return 0.0
    cv = np.std(sizes) / mean
    return round(max(0.0, 1.0 - cv), 4)


def _boundary_positions(chunks: List[Dict]) -> List[int]:
    """Return sentence-level start indices of each chunk (skip index 0)."""
    positions = []
    cumulative = 0
    for i, c in enumerate(chunks):
        if i > 0:
            positions.append(cumulative)
        cumulative += c.get("sentence_count", 1)
    return positions


def _total_sentences(chunks: List[Dict]) -> int:
    return sum(c.get("sentence_count", 1) for c in chunks)


def score_chunks(
    chunks: List[Dict],
    tagger: ConceptTagger,
    gold_boundaries: Optional[List[int]] = None,
) -> Dict:
    if not chunks:
        return {}

    coh  = chunk_coherence_score(chunks, tagger)
    pur  = concept_purity(chunks, tagger)
    con  = _concept_contrast(chunks)
    bal  = _size_balance(chunks)
    comp = round(0.35 * coh + 0.35 * pur + 0.20 * con + 0.10 * bal, 4)
    n    = len(chunks)
    avg  = round(np.mean([c.get("sentence_count", 1) for c in chunks]), 1)

    result = {
        "coherence":  round(coh, 3),
        "purity":     round(pur, 3),
        "contrast":   round(con, 3),
        "balance":    round(bal, 3),
        "composite":  comp,
        "n_chunks":   n,
        "avg_sents":  avg,
    }

    if gold_boundaries is not None:
        pred = _boundary_positions(chunks)
        n_sents = _total_sentences(chunks)
        p, r, f = boundary_precision_recall_f1(pred, gold_boundaries, n_sents, window=1)
        result["boundary_P"] = p
        result["boundary_R"] = r
        result["boundary_F1"] = f

    return result


# ---------------------------------------------------------------------------
# Build chunkers
# ---------------------------------------------------------------------------

def build_chunkers(
    run_semantic: bool,
    run_hybrid: bool,
    shared_embedder=None,
) -> List[Tuple[str, object]]:
    chunkers = []

    chunkers.append(("ontology-kw", OntologyChunker(
        shift_threshold=0.5,
        window_size=5,
        min_sentences=2,
        max_sentences=25,
        use_ensemble=False,
        confidence_drop_threshold=0.2,
    )))

    if run_hybrid:
        chunkers.append(("ontology-hyb", OntologyChunker(
            shift_threshold=0.5,
            window_size=5,
            min_sentences=2,
            max_sentences=25,
            use_ensemble=True,
            ontology_weight=0.5,
            confidence_drop_threshold=0.2,
            embedder=shared_embedder,
        )))

    if run_semantic:
        try:
            from src.baselines.semantic_chunker import SemanticChunker
            chunkers.append(("semantic", SemanticChunker(threshold=0.5, min_sentences=2)))
        except Exception as e:
            print(f"  [skip] semantic chunker: {e}")

    chunkers.append(("fixed-500",  FixedChunker(chunk_size=500,  overlap=50)))
    chunkers.append(("fixed-200",  FixedChunker(chunk_size=200,  overlap=20)))
    chunkers.append(("recursive",  RecursiveChunker(chunk_size=800, overlap=80)))

    return chunkers


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_results(
    label: str,
    results: Dict[str, Dict],
    has_boundaries: bool,
) -> None:
    names = list(results.keys())
    if not names:
        return

    has_b = has_boundaries and any("boundary_F1" in v for v in results.values())

    # Column definitions
    cols = ["coherence", "purity", "contrast", "balance", "composite",
            "n_chunks", "avg_sents"]
    if has_b:
        cols += ["boundary_P", "boundary_R", "boundary_F1"]

    col_w = 10
    name_w = 14

    header = f"  {'chunker':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep    = "  " + "-" * len(header.strip())

    print(f"\n{'='*70}")
    print(f"  File: {label}")
    print(f"{'='*70}")
    print(header)
    print(sep)

    # Sort by composite descending
    sorted_names = sorted(names, key=lambda n: results[n].get("composite", 0), reverse=True)

    for name in sorted_names:
        r = results[name]
        row = f"  {name:<{name_w}}"
        for c in cols:
            val = r.get(c, "-")
            if isinstance(val, float):
                row += f"{val:>{col_w}.3f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)

    print(sep)
    best = sorted_names[0]
    print(f"  Best: {best}  (composite={results[best]['composite']:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chunker benchmark")
    parser.add_argument("files", nargs="+", help="Arabic .txt/.test files to benchmark")
    parser.add_argument(
        "--boundaries",
        default=None,
        help='Comma-separated reference boundary positions, e.g. "6,11,15"'
             ' (sentence indices where new chunks START, 1-based)',
    )
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip SemanticChunker (no model download)")
    parser.add_argument("--no-hybrid",   action="store_true",
                        help="Skip OntologyChunker hybrid mode (no model download)")
    args = parser.parse_args()

    # Parse gold boundaries (convert to 0-based)
    gold: Optional[List[int]] = None
    if args.boundaries:
        gold = [int(x.strip()) - 1 for x in args.boundaries.split(",") if x.strip()]

    # Shared resources
    tagger = ConceptTagger(use_morphology=False)

    # Load embedding models ONCE if hybrid or semantic modes are active
    shared_embedder = None
    if not args.no_hybrid:
        try:
            from src.chunker.ensemble_embedder import EnsembleEmbedder
            print("Loading embedding models (once for all files) ...")
            shared_embedder = EnsembleEmbedder()
            shared_embedder._load_all()
            print("Models ready.\n")
        except Exception as e:
            print(f"  [warn] Could not load ensemble embedder: {e}")
            print("  Hybrid mode will be skipped.\n")

    chunkers = build_chunkers(
        run_semantic=not args.no_semantic,
        run_hybrid=not args.no_hybrid,
        shared_embedder=shared_embedder,
    )

    # Run on each file
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"[skip] File not found: {filepath}")
            continue

        text = path.read_text(encoding="utf-8")
        file_results: Dict[str, Dict] = {}

        for name, chunker in chunkers:
            try:
                chunks = chunker.chunk_dicts(text)
                file_results[name] = score_chunks(chunks, tagger, gold)
            except Exception as e:
                print(f"  [error] {name}: {e}")
                file_results[name] = {"composite": -1.0}

        _print_results(path.name, file_results, has_boundaries=gold is not None)

    print()


if __name__ == "__main__":
    main()

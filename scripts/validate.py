"""
Validate the OntologyChunker on the project's own data files.

What this script does
---------------------
1. Runs OntologyChunker on all 5 data files and prints every chunk
   with its concept, confidence, sentence count, and boundary reason.

2. Shows the concept distribution per file (which topics were detected
   and how many chunks each got).

3. Identifies "knowledge gaps" — windows where the tagger returned
   "عام / General" (confidence = 0.0), meaning no domain JSON matched.
   These are words/phrases you should add to the relevant domain file.

4. QA coverage test — for each of the 20 QA pairs in qa_pairs.json,
   checks whether the answer string is contained within a single chunk.
   - PASS: the relevant information was not split across a boundary.
   - FAIL: the chunker cut the text in the middle of the relevant passage.
   This is a zero-setup accuracy signal that uses only the existing files.

5. Per-file summary: number of chunks, avg sentences/chunk, % chunks
   with a non-general concept, % QA pairs that PASS.

Usage
-----
    cd arabic-ontology-chunker
    python scripts/validate.py [--verbose] [--file <filename>]

Options
-------
    --verbose     Print full chunk text, not just previews.
    --file        Run on a single file only (e.g. news/article_1.txt).
    --threshold   Shift threshold for OntologyChunker (default 0.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker.ontology_chunker import OntologyChunker, OntologyChunk
from src.chunker.concept_tagger import ConceptTagger
from src.preprocessing.normalizer import normalize as _normalize_text

DATA_DIR = ROOT / "data"

ALL_FILES = {
    "news/article_1.txt":              DATA_DIR / "news" / "article_1.txt",
    "news/article_2.txt":              DATA_DIR / "news" / "article_2.txt",
    "news/article_3.txt":              DATA_DIR / "news" / "article_3.txt",
    "islamic/quran_tafsir_sample.txt": DATA_DIR / "islamic" / "quran_tafsir_sample.txt",
    "islamic/hadith_sample.txt":       DATA_DIR / "islamic" / "hadith_sample.txt",
}

# ANSI colours (safe to use in any terminal; stripped if output is piped)
import os
_USE_COLOUR = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, 'fileno') else False

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

GREEN  = lambda t: _c(t, "32")
YELLOW = lambda t: _c(t, "33")
RED    = lambda t: _c(t, "31")
BOLD   = lambda t: _c(t, "1")
DIM    = lambda t: _c(t, "2")


# ---------------------------------------------------------------------------
# Section 1: chunk a single file and print results
# ---------------------------------------------------------------------------

def validate_file(
    file_key: str,
    text: str,
    chunker: OntologyChunker,
    qa_pairs: List[Dict],
    verbose: bool = False,
) -> Dict:
    """
    Chunk one file, print results, return summary stats.
    """
    print("\n" + "═" * 70)
    print(BOLD(f"  FILE: {file_key}"))
    print("═" * 70)

    chunks = chunker.chunk_rich(text)

    # ── Section 1: chunks ──────────────────────────────────────────────
    print(BOLD(f"\n  {len(chunks)} chunk(s) produced:\n"))
    gap_windows: List[str] = []

    for c in chunks:
        concept_label = (
            GREEN(f"{c.concept} / {c.concept_en}")
            if c.concept != "عام"
            else RED("عام / General  ← knowledge gap")
        )
        print(
            f"  Chunk {c.chunk_index:>2}  [{concept_label}]  "
            f"conf={c.confidence:.2f}  sents={c.sentence_count}"
        )
        print(DIM(f"           reason : {c.boundary_reason}"))

        preview = (c.text if verbose else c.text[:120].replace("\n", " ") + "…")
        print(f"           text   : {preview}")
        print()

        if c.concept == "عام":
            gap_windows.append(c.text[:80])

    # ── Section 2: concept distribution ────────────────────────────────
    concept_counts: Dict[str, int] = defaultdict(int)
    for c in chunks:
        concept_counts[f"{c.concept} ({c.concept_en})"] += 1

    print(BOLD("  Concept distribution:"))
    for label, count in sorted(concept_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {label:<30} {bar} {count}")

    # ── Section 3: knowledge gaps ───────────────────────────────────────
    if gap_windows:
        print(BOLD(f"\n  {RED(str(len(gap_windows)))} knowledge gap chunk(s) — "
                   f"add their keywords to the relevant domain JSON:\n"))
        for i, snippet in enumerate(gap_windows, 1):
            print(f"    [{i}] {snippet}…")
    else:
        print(GREEN("\n  No knowledge gaps — all chunks matched a domain."))

    # ── Section 4: QA coverage test ────────────────────────────────────
    file_qas = [qa for qa in qa_pairs if qa["source_file"] == file_key]
    if file_qas:
        print(BOLD(f"\n  QA Coverage test ({len(file_qas)} pairs):"))
        passes, fails = 0, 0
        for qa in file_qas:
            # Normalize the answer the same way the chunker normalizes text
            # so diacritics and alef variants don't cause false mismatches
            answer_norm = _normalize_text(qa["answer"]).lower()
            answer_lower = qa["answer"].lower()
            # Check if any single chunk contains the answer
            containing = [
                c for c in chunks
                if answer_norm in c.text.lower() or answer_lower in c.text.lower()
            ]
            if containing:
                passes += 1
                status = GREEN("PASS")
                detail = f"chunk {containing[0].chunk_index} [{containing[0].concept}]"
            else:
                fails += 1
                status = RED("FAIL")
                detail = "answer spans a boundary or not in text"

            print(f"    [{status}] Q: {qa['question'][:55]}…")
            print(f"            A: {qa['answer'][:55]}")
            print(DIM(f"            → {detail}\n"))

        pct = 100 * passes / len(file_qas)
        colour = GREEN if pct >= 75 else (YELLOW if pct >= 50 else RED)
        print(colour(f"    Coverage: {passes}/{len(file_qas)} ({pct:.0f}%)"))

    # ── Summary dict ────────────────────────────────────────────────────
    labelled = sum(1 for c in chunks if c.concept != "عام")
    avg_sents = sum(c.sentence_count for c in chunks) / max(len(chunks), 1)
    qa_pass = sum(
        1 for qa in file_qas
        if any(
            _normalize_text(qa["answer"]).lower() in c.text.lower()
            or qa["answer"].lower() in c.text.lower()
            for c in chunks
        )
    )
    return {
        "chunks": len(chunks),
        "avg_sentences_per_chunk": round(avg_sents, 1),
        "labelled_pct": round(100 * labelled / max(len(chunks), 1), 1),
        "knowledge_gaps": len(gap_windows),
        "qa_pass": qa_pass,
        "qa_total": len(file_qas),
        "qa_pct": round(100 * qa_pass / max(len(file_qas), 1), 1),
    }


# ---------------------------------------------------------------------------
# Overall summary table
# ---------------------------------------------------------------------------

def print_summary(summaries: Dict[str, Dict]) -> None:
    print("\n" + "═" * 80)
    print(BOLD("  OVERALL SUMMARY"))
    print("═" * 80)

    cols   = ["File", "Chunks", "AvgSents", "Labelled%", "Gaps", "QA%"]
    widths = [36, 7, 9, 10, 5, 6]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(BOLD(header))
    print("  " + "  ".join("-" * w for w in widths))

    for key, s in summaries.items():
        qa_str = f"{s['qa_pct']:.0f}%" if s['qa_total'] > 0 else "n/a"
        row = [
            key,
            str(s["chunks"]),
            str(s["avg_sentences_per_chunk"]),
            f"{s['labelled_pct']:.0f}%",
            str(s["knowledge_gaps"]),
            qa_str,
        ]
        line = "  ".join(v.ljust(w) for v, w in zip(row, widths))
        print(line)

    print("═" * 80)
    print()
    print(BOLD("  What to do with the results:"))
    print()
    print("  Labelled% < 80%  → domain JSONs are missing vocabulary.")
    print("                     Run with --verbose to see gap chunks,")
    print("                     then add keywords/roots to data/domains/.")
    print()
    print("  QA% < 75%        → chunker is splitting relevant passages.")
    print("                     Try lowering --threshold (e.g. 0.4) to")
    print("                     require higher confidence before splitting.")
    print()
    print("  Many 1-sentence  → min_sentences too low. Increase it.")
    print("  chunks           ")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OntologyChunker on data files.")
    parser.add_argument("--verbose",   action="store_true", help="Print full chunk text.")
    parser.add_argument("--file",      default=None,        help="Validate a single file key.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Shift threshold for OntologyChunker (default 0.5).")
    args = parser.parse_args()

    # Load QA pairs
    qa_path = DATA_DIR / "qa_pairs.json"
    qa_pairs = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else []

    # Build chunker
    chunker = OntologyChunker(shift_threshold=args.threshold, min_sentences=2, max_sentences=25)

    # Select files to validate
    files_to_run = ALL_FILES
    if args.file:
        files_to_run = {k: v for k, v in ALL_FILES.items() if args.file in k}
        if not files_to_run:
            print(f"No file matching '{args.file}'.  Available keys:")
            for k in ALL_FILES:
                print(f"  {k}")
            sys.exit(1)

    print(BOLD("\n  Arabic Ontology Chunker — Validation"))
    print(f"  threshold={args.threshold}  files={len(files_to_run)}\n")

    summaries: Dict[str, Dict] = {}
    for key, path in files_to_run.items():
        if not path.exists():
            print(f"  [skip] {key} — file not found at {path}")
            continue
        text = path.read_text(encoding="utf-8")
        summaries[key] = validate_file(key, text, chunker, qa_pairs, verbose=args.verbose)

    if len(summaries) > 1:
        print_summary(summaries)


if __name__ == "__main__":
    main()

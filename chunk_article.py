"""
Read an Arabic text file and print how the chunker split it.

Usage
-----
    .venv/bin/python chunk_article.py data/news/article_1.txt
    .venv/bin/python chunk_article.py data/islamic/hadith_sample.txt --threshold 0.4
    .venv/bin/python chunk_article.py my_article.txt --full
    .venv/bin/python chunk_article.py my_article.txt --keyword-mode   # pure ontology, no models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunker.ontology_chunker import OntologyChunker


def chunk_and_print(
    filepath: str,
    threshold: float = 0.5,
    full_text: bool = False,
    keyword_mode: bool = False,
):
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    chunker = OntologyChunker(
        shift_threshold=threshold,
        min_sentences=2,
        max_sentences=25,
        use_ensemble=not keyword_mode,
    )

    chunks = chunker.chunk_rich(text)

    mode = "keyword (ontology only)" if keyword_mode else "ensemble (3 models + ontology)"
    print(f"\nFile     : {path.name}")
    print(f"Mode     : {mode}")
    print(f"Threshold: {threshold}")
    print(f"Chunks   : {len(chunks)}")
    print("=" * 70)

    for c in chunks:
        print(f"\nChunk {c.chunk_index}  [{c.concept} / {c.concept_en}]")
        print(f"  Sentences : {c.sentence_count}")
        print(f"  Confidence: {c.confidence:.2f}")
        print(f"  Split reason: {c.boundary_reason}")
        print(f"  Keywords : {c.keywords[:6]}")
        print()
        if full_text:
            print(c.text)
        else:
            # Print each sentence on its own line for clarity
            for i, s in enumerate(c.sentences):
                print(f"  [{i+1}] {s}")
        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to Arabic .txt file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Boundary sensitivity (default 0.5, lower = fewer cuts)")
    parser.add_argument("--full", action="store_true",
                        help="Print full chunk text instead of sentence list")
    parser.add_argument("--keyword-mode", action="store_true",
                        help="Pure ontology mode: no embedding models, boundaries driven by concept shifts only")
    args = parser.parse_args()

    chunk_and_print(args.file, args.threshold, args.full, args.keyword_mode)

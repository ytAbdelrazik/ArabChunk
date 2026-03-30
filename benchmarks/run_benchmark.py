"""
Benchmark: OntologyChunker vs. Fixed / Recursive / Semantic baselines.

Chunkers tested (6 rows)
------------------------
  OntologyChunker          — ontology keyword + morphology (the main system)
  FixedChunker             — fixed token windows, no linguistics
  RecursiveChunker         — hierarchical separator splits
  Semantic-SBERT100K       — akhooli/Arabic-SBERT-100K   boundary detection
  Semantic-MatryoshkaFull  — Arabic-Triplet-Matryoshka-V2 (768-dim)
  Semantic-Matryoshka64    — Arabic-Triplet-Matryoshka-V2 truncated to 64-dim

Retrieval embedder (FAISS)
--------------------------
  Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2 (768-dim)
  Used for ALL chunkers — fair comparison of chunking quality, not embedding quality.
  Override with EMBED_MODEL env var.

Usage
-----
    cd arabic-ontology-chunker
    python -m benchmarks.run_benchmark

Options (environment variables)
--------------------------------
    EMBED_MODEL   Retrieval embedding model (default: Matryoshka-V2)
    SKIP_SEMANTIC Set to "1" to skip all SemanticChunker variants
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make project root importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker.ontology_chunker import OntologyChunker
from src.chunker.concept_tagger import ConceptTagger
from src.baselines.fixed_chunker import FixedChunker
from src.baselines.recursive_chunker import RecursiveChunker
from src.baselines.semantic_chunker import SemanticChunker, SBERT_100K, MATRYOSHKA
from src.evaluation.metrics import (
    EvaluationReport,
    precision_at_k,
    chunk_coherence_score,
    concept_purity,
)

DATA_DIR   = ROOT / "data"
RESULTS_PATH = ROOT / "benchmarks" / "results.json"

EMBED_MODEL   = os.environ.get("EMBED_MODEL", MATRYOSHKA)   # best Arabic retrieval model
SKIP_SEMANTIC = os.environ.get("SKIP_SEMANTIC", "0") == "1"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class ArabertEmbedder:
    """
    Sentence embedder backed by aubmindlab/bert-base-arabertv02.

    Uses transformers directly with mean pooling so we are not constrained
    by sentence-transformers' model registry.  Falls back to the multilingual
    MiniLM if the Arabic model fails to load.
    """

    FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._dim: int = 768

    def _load(self) -> None:
        if self._model is not None:
            return

        # --- Try transformers (primary) ---
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            print(f"[embedder] Loading {self.model_name} …", flush=True)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._backend = "transformers"
            self._dim = self._model.config.hidden_size
            print(f"[embedder] Loaded ({self._dim}-dim)", flush=True)
            return
        except Exception as e:
            print(f"[embedder] Primary model failed ({e}), trying fallback…")

        # --- Fallback: sentence-transformers with MiniLM ---
        try:
            from sentence_transformers import SentenceTransformer

            print(f"[embedder] Loading fallback {self.FALLBACK} …", flush=True)
            self._model = SentenceTransformer(self.FALLBACK)
            self._backend = "sbert"
            self._dim = self._model.get_sentence_embedding_dimension()
            print(f"[embedder] Fallback loaded ({self._dim}-dim)", flush=True)
        except ImportError as exc:
            raise RuntimeError(
                "No embedding backend available.\n"
                "Install at least one of:\n"
                "  pip install transformers torch\n"
                "  pip install sentence-transformers"
            ) from exc

    def _mean_pool(
        self, token_embeds: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        import torch
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeds * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, dim)."""
        self._load()

        if self._backend == "sbert":
            embs = self._model.encode(
                texts, batch_size=batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            return embs.astype(np.float32)

        # transformers path
        import torch

        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                out = self._model(**enc)
            emb = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_embs.append(emb.numpy().astype(np.float32))

        mat = np.vstack(all_embs) if all_embs else np.zeros((0, self._dim), np.float32)
        # L2 normalise for cosine via inner product
        norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
        return mat / norms

    @property
    def dim(self) -> int:
        self._load()
        return self._dim


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

class FAISSIndex:
    """Flat inner-product index (cosine similarity after L2 norm)."""

    def __init__(self, dim: int) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(dim)
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for benchmarking.\n"
                "Install with: pip install faiss-cpu"
            ) from exc
        self._chunks: List[Dict] = []

    def add(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        import faiss
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, k: int = 3) -> List[Dict]:
        k = min(k, len(self._chunks))
        if k == 0:
            return []
        scores, idxs = self._index.search(query_emb.reshape(1, -1), k)
        return [self._chunks[i] for i in idxs[0] if i >= 0]

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_files() -> Dict[str, str]:
    """Return {relative_path: text} for all 5 source files."""
    files = {
        "news/article_1.txt":              DATA_DIR / "news" / "article_1.txt",
        "news/article_2.txt":              DATA_DIR / "news" / "article_2.txt",
        "news/article_3.txt":              DATA_DIR / "news" / "article_3.txt",
        "islamic/quran_tafsir_sample.txt": DATA_DIR / "islamic" / "quran_tafsir_sample.txt",
        "islamic/hadith_sample.txt":       DATA_DIR / "islamic" / "hadith_sample.txt",
    }
    return {
        key: path.read_text(encoding="utf-8")
        for key, path in files.items()
        if path.exists()
    }


def load_qa_pairs() -> List[Dict]:
    path = DATA_DIR / "qa_pairs.json"
    if not path.exists():
        raise FileNotFoundError(f"QA pairs not found at {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Per-chunker benchmark
# ---------------------------------------------------------------------------

def build_indices(
    chunker_name: str,
    chunker,
    documents: Dict[str, str],
    embedder: ArabertEmbedder,
) -> Dict[str, FAISSIndex]:
    """
    Chunk every document and build one FAISS index per file.

    Returns
    -------
    {source_file: FAISSIndex}
    """
    indices: Dict[str, FAISSIndex] = {}
    for file_key, text in documents.items():
        chunks = chunker.chunk_dicts(text)
        if not chunks:
            continue
        texts = [c["text"] for c in chunks]
        embs = embedder.encode(texts)
        idx = FAISSIndex(dim=embedder.dim)
        idx.add(chunks, embs)
        indices[file_key] = idx
        print(
            f"  [{chunker_name}] {file_key}: "
            f"{len(chunks)} chunks, index={len(idx)}"
        )
    return indices


def run_qa_eval(
    qa_pairs: List[Dict],
    indices: Dict[str, FAISSIndex],
    embedder: ArabertEmbedder,
    k: int = 3,
) -> Tuple[float, float]:
    """
    Evaluate a chunker's indices on all QA pairs.

    Returns
    -------
    (avg_precision_at_1, avg_precision_at_3)
    """
    p1_scores: List[float] = []
    p3_scores: List[float] = []

    for qa in qa_pairs:
        source = qa["source_file"]
        answer = qa["answer"]
        question = qa["question"]

        idx = indices.get(source)
        if idx is None or len(idx) == 0:
            p1_scores.append(0.0)
            p3_scores.append(0.0)
            continue

        q_emb = embedder.encode([question])
        top_chunks = idx.search(q_emb, k=k)

        p1_scores.append(precision_at_k(top_chunks, answer, k=1))
        p3_scores.append(precision_at_k(top_chunks, answer, k=3))

    avg_p1 = round(float(np.mean(p1_scores)), 4) if p1_scores else 0.0
    avg_p3 = round(float(np.mean(p3_scores)), 4) if p3_scores else 0.0
    return avg_p1, avg_p3


def compute_quality_metrics(
    chunker_name: str,
    chunker,
    documents: Dict[str, str],
    tagger: ConceptTagger,
) -> Tuple[float, float]:
    """
    Compute avg chunk_coherence_score and concept_purity across all documents.

    Returns
    -------
    (avg_coherence, avg_purity)
    """
    coherence_scores: List[float] = []
    purity_scores: List[float] = []

    for text in documents.values():
        chunks = chunker.chunk_dicts(text)
        if not chunks:
            continue
        coh = chunk_coherence_score(chunks, tagger)
        pur = concept_purity(chunks, tagger)
        if not (coh != coh):  # not NaN
            coherence_scores.append(coh)
        if not (pur != pur):
            purity_scores.append(pur)

    avg_coh = round(float(np.mean(coherence_scores)), 4) if coherence_scores else float("nan")
    avg_pur = round(float(np.mean(purity_scores)), 4) if purity_scores else float("nan")
    return avg_coh, avg_pur


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_table(results: Dict[str, Dict]) -> None:
    cols = ["Chunker", "P@1", "P@3", "Coherence", "Purity", "Time(s)"]
    widths = [26, 7, 7, 11, 8, 9]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)

    print("\n" + "=" * len(header))
    print("  BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, r in results.items():
        row = [
            name,
            f"{r['precision_at_1']:.4f}",
            f"{r['precision_at_3']:.4f}",
            f"{r['chunk_coherence']:.4f}" if r['chunk_coherence'] == r['chunk_coherence'] else "  n/a  ",
            f"{r['concept_purity']:.4f}"  if r['concept_purity'] == r['concept_purity']  else "  n/a  ",
            f"{r['elapsed_s']:.1f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Arabic Ontology Chunker — Benchmark")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data files …")
    documents = load_data_files()
    qa_pairs  = load_qa_pairs()
    print(f"  {len(documents)} documents, {len(qa_pairs)} QA pairs")

    # Embedding model
    print(f"\n[2/5] Initialising embedder ({EMBED_MODEL}) …")
    embedder = ArabertEmbedder(EMBED_MODEL)
    embedder._load()  # warm-up

    # Shared tagger for quality metrics
    tagger = ConceptTagger()

    # Chunker registry
    # OntologyChunker is always run — it is the main system being evaluated.
    # The other four are baselines for comparison only.
    chunkers: Dict[str, object] = {
        "OntologyChunker":        OntologyChunker(shift_threshold=0.6, min_sentences=2),
        "FixedChunker":           FixedChunker(chunk_size=500, overlap=50),
        "RecursiveChunker":       RecursiveChunker(chunk_size=500, overlap=50),
    }
    if not SKIP_SEMANTIC:
        chunkers["Semantic-SBERT100K"]      = SemanticChunker(
            model_name=SBERT_100K, threshold=0.5
        )
        chunkers["Semantic-Matryoshka768"]  = SemanticChunker(
            model_name=MATRYOSHKA, threshold=0.5
        )
        chunkers["Semantic-Matryoshka64"]   = SemanticChunker(
            model_name=MATRYOSHKA, threshold=0.5, matryoshka_dim=64
        )

    all_results: Dict[str, Dict] = {}

    print(f"\n[3/5] Running {len(chunkers)} chunkers …\n")
    for name, chunker in chunkers.items():
        print(f"── {name}")
        t0 = time.time()

        # Build FAISS indices
        indices = build_indices(name, chunker, documents, embedder)

        # QA evaluation
        avg_p1, avg_p3 = run_qa_eval(qa_pairs, indices, embedder)

        # Quality metrics
        avg_coh, avg_pur = compute_quality_metrics(name, chunker, documents, tagger)

        elapsed = round(time.time() - t0, 2)
        all_results[name] = {
            "precision_at_1": avg_p1,
            "precision_at_3": avg_p3,
            "chunk_coherence": avg_coh,
            "concept_purity":  avg_pur,
            "elapsed_s":       elapsed,
        }
        print(
            f"  P@1={avg_p1:.4f}  P@3={avg_p3:.4f}  "
            f"coherence={avg_coh:.4f}  purity={avg_pur:.4f}  "
            f"time={elapsed}s\n"
        )

    # Print table
    print_table(all_results)

    # Save JSON
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print(f"\n[5/5] Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()

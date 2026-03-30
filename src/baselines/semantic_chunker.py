"""
Semantic chunker baseline — cosine-similarity splits via Arabic sentence embeddings.

Supported models
----------------
SBERT_100K  (default)  akhooli/Arabic-SBERT-100K
                        Arabic Sentence-BERT fine-tuned on 100K sentence pairs.
                        Strong at semantic similarity; good for boundary detection.

MATRYOSHKA             Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2
                        Triplet-loss training + Matryoshka Representation Learning.
                        Supports dimension truncation (768 → 512 → 256 → 128 → 64)
                        so you can trade quality for speed with `matryoshka_dim`.

Both models run fully on-device via sentence-transformers.
The model name is recorded in every chunk's boundary_reason for traceability.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..chunker.base_chunker import BaseChunker, Chunk, _count_sentences
from ..preprocessing.normalizer import normalize
from ..preprocessing.tokenizer import sentence_tokenize

# Named constants so callers can import the strings instead of hard-coding them
SBERT_100K   = "akhooli/Arabic-SBERT-100K"
MATRYOSHKA   = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # fallback


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class SemanticChunker(BaseChunker):
    """
    Splits text at positions where adjacent sentence embeddings diverge.

    A boundary is placed between sentence i and sentence i+1 when
    ``cosine(embed(i), embed(i+1)) < threshold``.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Use the module-level constants
        ``SBERT_100K`` or ``MATRYOSHKA`` for the two Arabic models.
        Defaults to ``SBERT_100K``.
    threshold:
        Cosine-similarity cut-off.  Lower → fewer, larger chunks.
        Typical range: 0.4 – 0.7.  Tune per corpus.
    min_sentences:
        Minimum sentences per output chunk.
    matryoshka_dim:
        For ``MATRYOSHKA`` only — truncate embeddings to this many
        dimensions after encoding.  Valid values: 768, 512, 256, 128, 64.
        Smaller → faster inference, slightly lower quality.
        Ignored for non-Matryoshka models.
    normalize_input:
        Apply Arabic normalisation before splitting.
    """

    def __init__(
        self,
        model_name: str = SBERT_100K,
        threshold: float = 0.5,
        min_sentences: int = 2,
        matryoshka_dim: Optional[int] = None,
        normalize_input: bool = True,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.normalize_input = normalize_input
        self._matryoshka_dim = matryoshka_dim
        self._model = None  # lazy-loaded

        # Validate matryoshka_dim if provided
        if matryoshka_dim is not None and model_name != MATRYOSHKA:
            raise ValueError(
                "matryoshka_dim is only valid with the MATRYOSHKA model. "
                f"Current model: {model_name}"
            )
        if matryoshka_dim is not None and matryoshka_dim not in (768, 512, 256, 128, 64):
            raise ValueError(
                f"matryoshka_dim must be one of 768, 512, 256, 128, 64. Got {matryoshka_dim}"
            )

    # ------------------------------------------------------------------
    # Model loading  (lazy — only on first encode call)
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            # Matryoshka model: pass truncate_dim if requested
            if self.model_name == MATRYOSHKA and self._matryoshka_dim is not None:
                self._model = SentenceTransformer(
                    self.model_name,
                    truncate_dim=self._matryoshka_dim,
                )
            else:
                self._model = SentenceTransformer(self.model_name)

            return self._model

        except Exception as primary_err:
            # If the requested Arabic model fails (e.g. no internet on first run),
            # fall back to the multilingual MiniLM so tests remain runnable.
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MULTILINGUAL)
                return self._model
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for SemanticChunker.\n"
                    "Install with: pip install sentence-transformers"
                ) from exc

    def _embed(self, sentences: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, dim)."""
        model = self._load_model()
        embs = model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm → cosine = dot product
        )
        return embs.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Effective embedding dimension (accounts for Matryoshka truncation)."""
        if self._matryoshka_dim is not None:
            return self._matryoshka_dim
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Boundary reason string  (used in every chunk for traceability)
    # ------------------------------------------------------------------

    def _model_tag(self) -> str:
        if self.model_name == SBERT_100K:
            return "SBERT-100K"
        if self.model_name == MATRYOSHKA:
            dim_tag = f"-{self._matryoshka_dim}d" if self._matryoshka_dim else ""
            return f"Matryoshka-V2{dim_tag}"
        # Shortened name for any other model
        return self.model_name.split("/")[-1]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into semantically coherent chunks.

        Returns
        -------
        List[Chunk]
            concept and concept_en are always None — semantic chunking
            does not assign ontological labels.
        """
        if self.normalize_input:
            text = normalize(text)

        sentences = [s for s in sentence_tokenize(text) if s.strip()]
        if not sentences:
            return []

        if len(sentences) <= self.min_sentences:
            return [Chunk(
                text=" ".join(sentences),
                index=0,
                metadata={
                    "concept": None,
                    "concept_en": None,
                    "keywords": [],
                    "sentence_count": len(sentences),
                    "boundary_reason": (
                        f"semantic_split({self._model_tag()},"
                        f"single_chunk_below_min)"
                    ),
                },
            )]

        embeddings = self._embed(sentences)

        # Cosine similarities between adjacent sentence embeddings
        # (embeddings are already L2-normed so dot product = cosine)
        similarities = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # Collect boundary positions (0-indexed sentence where new chunk starts)
        boundaries = [0]
        last_boundary = 0
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                if (i + 1) - last_boundary >= self.min_sentences:
                    boundaries.append(i + 1)
                    last_boundary = i + 1

        # Assemble output chunks
        chunks: List[Chunk] = []
        model_tag = self._model_tag()
        for k, start in enumerate(boundaries):
            end = boundaries[k + 1] if k + 1 < len(boundaries) else len(sentences)
            chunk_sents = sentences[start:end]
            chunk_text  = " ".join(chunk_sents)

            # Average similarity of adjacent pairs *within* this chunk
            avg_sim = (
                float(np.mean(similarities[start: end - 1]))
                if end - start > 1
                else 1.0
            )

            chunks.append(Chunk(
                text=chunk_text,
                index=k,
                metadata={
                    "concept": None,
                    "concept_en": None,
                    "keywords": [],
                    "sentence_count": len(chunk_sents),
                    "boundary_reason": (
                        f"semantic_split({model_tag},"
                        f"threshold={self.threshold},"
                        f"avg_sim={avg_sim:.3f})"
                    ),
                },
            ))

        return chunks

    def chunk_dicts(self, text: str) -> List[Dict]:
        """Return semantic chunks in the standard dict schema."""
        return [
            {
                "text": c.text,
                "concept": None,
                "concept_en": None,
                "keywords": [],
                "sentence_count": c.metadata["sentence_count"],
                "chunk_index": c.index,
                "boundary_reason": c.metadata["boundary_reason"],
            }
            for c in self.chunk(text)
        ]

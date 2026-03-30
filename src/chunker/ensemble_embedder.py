"""
Ensemble Arabic sentence embedder.

Loads three Arabic-specialised models, encodes text with each, L2-normalises
each model's output, then averages across models to produce a single ensemble
embedding per sentence/window.

Models
------
SBERT_100K    akhooli/Arabic-SBERT-100K
              Arabic Sentence-BERT fine-tuned on 100K sentence pairs.
              Strongest at semantic similarity.

MATRYOSHKA    Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2
              Triplet-loss training, Matryoshka dims (768/512/256/128/64).
              Best at ranking and retrieval.

ARABERT       aubmindlab/bert-base-arabertv02
              Standard Arabic BERT with mean-pooling applied at encode time.
              Captures deep morphological patterns.

The three models see different aspects of Arabic text; averaging their
L2-normalised embeddings cancels out individual weaknesses and produces
more robust representations than any single model alone.

All models are lazy-loaded on first encode() call so importing this
module is always free.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


# Public model name constants (import instead of hard-coding strings)
SBERT_100K  = "akhooli/Arabic-SBERT-100K"
MATRYOSHKA  = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
ARABERT     = "aubmindlab/bert-base-arabertv02"

DEFAULT_MODELS = [SBERT_100K, MATRYOSHKA, ARABERT]

# Dimension produced by all three models
_DIM = 768


def _l2_norm(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
    return mat / norms


def _load_sbert(model_name: str):
    """Load a standard sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _load_arabert(model_name: str):
    """Load AraBERT via the sentence-transformers Transformer+Pooling modules."""
    from sentence_transformers import SentenceTransformer, models as st_models
    word_model = st_models.Transformer(model_name, max_seq_length=512)
    pool_model = st_models.Pooling(
        word_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
    )
    return SentenceTransformer(modules=[word_model, pool_model])


def _encode_one(model, texts: List[str]) -> np.ndarray:
    """Encode with one model, return L2-normalised float32 array."""
    embs = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 norm per model before averaging
    )
    return embs.astype(np.float32)


class EnsembleEmbedder:
    """
    Three-model Arabic ensemble embedder.

    Parameters
    ----------
    models:
        List of HuggingFace model names to ensemble.
        Defaults to [SBERT_100K, MATRYOSHKA, ARABERT].
    weights:
        Per-model weights for the weighted average.  Must sum to 1.0.
        Defaults to equal weights (1/N each).
    batch_size:
        Encoding batch size per model.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        batch_size: int = 32,
    ) -> None:
        self._model_names: List[str] = models if models is not None else DEFAULT_MODELS
        self._batch_size = batch_size
        self._loaded: List = []          # populated on first encode()
        self._failed: List[str] = []     # models that failed to load

        n = len(self._model_names)
        if weights is not None:
            if len(weights) != n:
                raise ValueError(
                    f"len(weights)={len(weights)} must equal len(models)={n}"
                )
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("weights must sum to 1.0")
            self._weights = list(weights)
        else:
            self._weights = [1.0 / n] * n   # equal weights

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all models (called once on first encode())."""
        if self._loaded:
            return

        for name in self._model_names:
            try:
                if name == ARABERT:
                    model = _load_arabert(name)
                else:
                    model = _load_sbert(name)
                self._loaded.append((name, model))
                print(f"[ensemble] loaded {name}", flush=True)
            except Exception as exc:
                self._failed.append(name)
                print(f"[ensemble] WARNING: could not load {name}: {exc}", flush=True)

        if not self._loaded:
            raise RuntimeError(
                "No embedding models could be loaded for EnsembleEmbedder.\n"
                "Install sentence-transformers: pip install sentence-transformers\n"
                "Then ensure the models are reachable from HuggingFace Hub."
            )

        # Re-normalise weights to the models that actually loaded
        loaded_names = {name for name, _ in self._loaded}
        active_weights = [
            w for name, w in zip(self._model_names, self._weights)
            if name in loaded_names
        ]
        total = sum(active_weights)
        self._active_weights = [w / total for w in active_weights]

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts and return ensemble embeddings.

        Steps
        -----
        1. Each loaded model encodes all texts → (N, 768) matrix.
        2. Each matrix is L2-normalised row-wise (unit sphere).
        3. Matrices are averaged with per-model weights.
        4. The result is L2-normalised again (optional but keeps cosine
           similarity numerically stable).

        Returns
        -------
        np.ndarray of shape (len(texts), 768), dtype float32.
        """
        self._load_all()

        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)

        stack: List[np.ndarray] = []
        for (_, model), weight in zip(self._loaded, self._active_weights):
            embs = _encode_one(model, texts)         # (N, 768), already L2-normed
            stack.append(embs * weight)

        ensemble = np.sum(stack, axis=0)             # weighted average
        return _l2_norm(ensemble)                    # re-normalise ensemble

    @property
    def dim(self) -> int:
        return _DIM

    @property
    def loaded_models(self) -> List[str]:
        """Names of successfully loaded models."""
        return [name for name, _ in self._loaded]

    @property
    def failed_models(self) -> List[str]:
        """Names of models that failed to load."""
        return list(self._failed)

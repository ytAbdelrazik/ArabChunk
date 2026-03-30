"""
Ontology-aware Arabic text chunker — fully offline.

Pipeline (ensemble mode, default)
----------------------------------
  1. Normalize (diacritics, tatweel, alef variants)
  2. Sentence tokenize
  3. Embed every sliding window with EnsembleEmbedder
     (3 Arabic models: SBERT-100K, Matryoshka-V2, AraBERT)
  4. Place a boundary where cosine similarity between consecutive
     windows drops below ``shift_threshold``
  5. Tag each assembled chunk with ConceptTagger (domain JSONs)
     to assign concept labels — entirely offline
  6. Chunk size enforcement (min / max sentences; merge small)

Pipeline (keyword mode, ``use_ensemble=False``)
------------------------------------------------
  Same as above but boundaries are detected by the ConceptTagger's
  ``detect_shift()`` method — no sentence-transformers required.
  Useful for unit tests, CI, or resource-constrained environments.

Every boundary carries a ``boundary_reason`` field that names which
layer triggered it (embedding cosine drop, concept shift, max-size
limit, etc.), making the system fully explainable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from .base_chunker import BaseChunker, Chunk
from .concept_tagger import ConceptResult, ConceptTagger
from ..preprocessing.normalizer import normalize
from ..preprocessing.tokenizer import sentence_tokenize


# ---------------------------------------------------------------------------
# Extended chunk type
# ---------------------------------------------------------------------------

@dataclass
class OntologyChunk:
    """
    A semantically coherent Arabic text chunk with full provenance.

    Attributes
    ----------
    text:           Reconstructed chunk text.
    concept:        Primary Arabic concept label.
    concept_en:     English gloss of the concept.
    confidence:     Tagger confidence for the dominant concept.
    keywords:       Key Arabic terms that triggered the concept.
    sentence_count: Number of sentences in this chunk.
    chunk_index:    Zero-based position in the output sequence.
    boundary_reason:Human-readable explanation of why the boundary
                    was placed here (which layer triggered it).
    sentences:      The individual sentence strings (for inspection).
    """

    text: str
    concept: str
    concept_en: str
    confidence: float
    keywords: List[str]
    sentence_count: int
    chunk_index: int
    boundary_reason: str
    sentences: List[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "concept": self.concept,
            "concept_en": self.concept_en,
            "confidence": self.confidence,
            "keywords": self.keywords,
            "sentence_count": self.sentence_count,
            "chunk_index": self.chunk_index,
            "boundary_reason": self.boundary_reason,
        }

    # Compatibility with BaseChunker.Chunk interface
    def to_base_chunk(self) -> Chunk:
        return Chunk(
            text=self.text,
            index=self.chunk_index,
            metadata=self.to_dict(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tag_window(
    tagger: ConceptTagger,
    sentences: List[str],
    i: int,
    window: int,
) -> ConceptResult:
    """Tag sentences[i : i+window], clamping to list bounds."""
    end = min(i + window, len(sentences))
    return tagger.tag(sentences[i:end])


def _build_chunk(
    sentences: List[str],
    concept: ConceptResult,
    index: int,
    reason: str,
) -> OntologyChunk:
    text = " ".join(sentences)
    return OntologyChunk(
        text=text,
        concept=concept.concept,
        concept_en=concept.concept_en,
        confidence=concept.confidence,
        keywords=concept.keywords,
        sentence_count=len(sentences),
        chunk_index=index,
        boundary_reason=reason,
        sentences=sentences,
    )


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class OntologyChunker(BaseChunker):
    """
    Splits Arabic text into ontologically coherent chunks.

    Fully offline — no API keys, no network. Same input always produces
    the same output. New domains are added by dropping a JSON file into
    ``data/domains/`` with no code changes required.

    Parameters
    ----------
    window_size:
        Number of sentences per tagging window (3–5 recommended).
    step:
        Sliding-window step size (1 = maximum sensitivity).
    shift_threshold:
        Boundary sensitivity.  In ensemble mode: minimum cosine-similarity
        drop to declare a boundary (lower → more boundaries).  In keyword
        mode: minimum incoming concept confidence to declare a boundary.
    min_sentences:
        Minimum sentences per output chunk (smaller chunks are merged).
    max_sentences:
        Maximum sentences per output chunk (oversized chunks are split).
    domains_dir:
        Override path to domain JSON files.
    normalize_input:
        Apply Arabic normalisation before processing.
    use_ensemble:
        If True (default) use EnsembleEmbedder (3 Arabic models) for
        boundary detection.  If False fall back to keyword-based
        ``ConceptTagger.detect_shift()`` — no model downloads required.
    ensemble_models:
        List of HuggingFace model IDs to use in the ensemble.
        Defaults to [SBERT-100K, Matryoshka-V2, AraBERT].
    ensemble_weights:
        Per-model weights.  Must sum to 1.0.  Defaults to equal weights.
    embedder:
        Optional pre-built EnsembleEmbedder to reuse across multiple
        OntologyChunker instances (e.g. during a parameter sweep).
        When supplied, the models are never re-loaded regardless of
        ``ensemble_models`` / ``ensemble_weights``.
    ontology_weight:
        Only used when ``use_ensemble=True``.  Controls how much the
        ontology signal contributes to the hybrid boundary score vs the
        pure embedding cosine dissimilarity.

        - 0.0  → pure embedding mode (original behaviour, no ontology).
        - 0.5  → equal weight: ontology and embeddings both contribute.
        - 1.0  → ontology only (same as ``use_ensemble=False`` but still
                  requires model loading for the embedding side).

        Boundary score = ontology_weight × ontology_signal
                       + (1 − ontology_weight) × embedding_dissimilarity

        A boundary is placed when score ≥ ``shift_threshold``.
        Recommended starting value: 0.5.
    confidence_drop_threshold:
        In keyword mode only.  When the ontology confidence drops by at
        least this amount between consecutive windows *within the same
        concept*, a boundary is placed.  Catches intra-domain topic
        drift (e.g. imagery → civilisation, both tagged as Literature).
        Set to 1.0 to disable intra-domain splitting.  Default: 0.35.
    """

    def __init__(
        self,
        window_size: int = 3,
        step: int = 1,
        shift_threshold: float = 0.7,
        min_sentences: int = 3,
        max_sentences: int = 15,
        domains_dir: Optional[Path] = None,
        normalize_input: bool = True,
        use_ensemble: bool = True,
        ensemble_models: Optional[List[str]] = None,
        ensemble_weights: Optional[List[float]] = None,
        ontology_weight: float = 0.5,
        confidence_drop_threshold: float = 0.35,
        embedder=None,
    ) -> None:
        if not (1 <= step <= window_size):
            raise ValueError("step must satisfy 1 <= step <= window_size")
        if min_sentences < 1:
            raise ValueError("min_sentences must be >= 1")
        if max_sentences < min_sentences:
            raise ValueError("max_sentences must be >= min_sentences")

        self.window_size = window_size
        self.step = step
        self.shift_threshold = shift_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.normalize_input = normalize_input
        self.use_ensemble = use_ensemble
        self.ontology_weight = max(0.0, min(1.0, ontology_weight))
        self.confidence_drop_threshold = confidence_drop_threshold

        self.tagger = ConceptTagger(domains_dir=domains_dir)

        # Ensemble embedder — use supplied instance or lazy-load on first use
        self._embedder = embedder
        self._ensemble_models = ensemble_models
        self._ensemble_weights = ensemble_weights

    # ------------------------------------------------------------------
    # Lazy ensemble embedder
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Return the EnsembleEmbedder, initialising it once."""
        if self._embedder is None:
            from .ensemble_embedder import EnsembleEmbedder
            self._embedder = EnsembleEmbedder(
                models=self._ensemble_models,
                weights=self._ensemble_weights,
            )
        return self._embedder

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, stripping empties."""
        return [s for s in sentence_tokenize(text) if s.strip()]

    # ------------------------------------------------------------------
    # Boundary detection — ensemble mode
    # ------------------------------------------------------------------

    def _find_boundaries_ensemble(
        self, sentences: List[str]
    ) -> List[tuple[int, str, ConceptResult]]:
        """
        Detect boundaries using EnsembleEmbedder cosine similarity.

        Steps
        -----
        1. Build one text string per sliding window.
        2. Encode all windows in a single batched call.
        3. Compute cosine similarity between consecutive window embeddings
           (embeddings are already L2-normalised → dot product = cosine).
        4. Boundary at position ``i + step`` when sim < shift_threshold
           AND the current chunk has ≥ min_sentences.
        5. After all boundaries are determined, tag each chunk's sentences
           with ConceptTagger for ontological labelling.
        """
        n = len(sentences)
        embedder = self._get_embedder()

        # Build window texts (one string per window position)
        window_positions = list(range(0, n, self.step))
        window_texts = []
        for pos in window_positions:
            end = min(pos + self.window_size, n)
            window_texts.append(" ".join(sentences[pos:end]))

        # Encode all windows in one batched call
        embeddings = embedder.encode(window_texts)   # (W, 768)

        # Cosine similarity between consecutive windows
        # (L2-normed → dot product = cosine)
        similarities = [
            float(np.dot(embeddings[k], embeddings[k + 1]))
            for k in range(len(embeddings) - 1)
        ]

        # ---- Build boundary list ----------------------------------------
        # boundary_starts: list of sentence-level start indices
        boundary_starts = [0]
        last_boundary = 0

        for k, sim in enumerate(similarities):
            # The window at index k+1 starts at sentence window_positions[k+1]
            sent_idx = window_positions[k + 1]
            chunk_len = sent_idx - last_boundary

            if sim < self.shift_threshold and chunk_len >= self.min_sentences:
                boundary_starts.append(sent_idx)
                last_boundary = sent_idx

        # ---- Tag each chunk for concept labels --------------------------
        boundaries: List[tuple[int, str, ConceptResult]] = []
        for b_idx, start in enumerate(boundary_starts):
            end = boundary_starts[b_idx + 1] if b_idx + 1 < len(boundary_starts) else n
            concept = self.tagger.tag(sentences[start:end])

            if b_idx == 0:
                reason = "start"
            else:
                # Find which similarity triggered this boundary
                # (the window-step transition just before `start`)
                k = window_positions.index(start) - 1
                sim_val = similarities[k] if 0 <= k < len(similarities) else 0.0
                reason = (
                    f"ensemble_cosine_drop: sim={sim_val:.3f} < "
                    f"threshold={self.shift_threshold} → "
                    f"'{concept.concept}' ({concept.concept_en})"
                )

            boundaries.append((start, reason, concept))

        return boundaries

    # ------------------------------------------------------------------
    # Boundary detection — hybrid mode (ontology + embeddings combined)
    # ------------------------------------------------------------------

    def _find_boundaries_hybrid(
        self, sentences: List[str]
    ) -> List[tuple[int, str, ConceptResult]]:
        """
        Hybrid boundary detection: ontology signal + embedding dissimilarity.

        For each consecutive window pair, computes:
            ontology_signal  — how strongly the concept changed (0→1)
            embed_signal     — cosine dissimilarity between window vectors (0→1)
            combined         — weighted average of the two signals

        A boundary is placed when combined >= shift_threshold.

        This means ontology and embeddings CONFIRM each other:
        - Strong ontology shift + high dissimilarity → very confident boundary.
        - One signal alone can still trigger a boundary if strong enough.
        """
        n = len(sentences)
        embedder = self._get_embedder()

        window_positions = list(range(0, n, self.step))
        window_texts = [
            " ".join(sentences[pos: min(pos + self.window_size, n)])
            for pos in window_positions
        ]

        # Encode all windows once — reused across all signal computations
        embeddings = embedder.encode(window_texts)   # (W, 768), L2-normed

        # Tag all windows once via ontology (hits cache on re-calls)
        ontology: List[ConceptResult] = [
            _tag_window(self.tagger, sentences, pos, self.window_size)
            for pos in window_positions
        ]

        # ---- Per-transition scoring ------------------------------------
        # Store events: (sent_idx, combined, ont_sig, emb_sig, curr_concept)
        boundary_events: List[tuple] = []
        last_boundary = 0

        for k in range(len(window_positions) - 1):
            sent_idx = window_positions[k + 1]
            chunk_len = sent_idx - last_boundary

            if chunk_len < self.min_sentences:
                continue

            # Embedding dissimilarity — cosine ∈ [-1,1], normalise to [0,1]
            sim = float(np.dot(embeddings[k], embeddings[k + 1]))
            embed_signal = max(0.0, (1.0 - sim) / 2.0)

            # Ontology shift signal
            prev_c, curr_c = ontology[k], ontology[k + 1]
            if prev_c.concept != curr_c.concept:
                # Cross-domain: strength = confidence of incoming concept
                ontology_signal = curr_c.confidence
            else:
                # Intra-domain: strength = confidence drop (if any)
                ontology_signal = max(0.0, prev_c.confidence - curr_c.confidence)

            combined = (
                self.ontology_weight * ontology_signal
                + (1.0 - self.ontology_weight) * embed_signal
            )

            if combined >= self.shift_threshold:
                boundary_events.append(
                    (sent_idx, combined, ontology_signal, embed_signal, curr_c)
                )
                last_boundary = sent_idx

        # ---- Build boundary list with reasons --------------------------
        boundary_starts = [0] + [ev[0] for ev in boundary_events]
        boundaries: List[tuple[int, str, ConceptResult]] = []

        for b_idx, start in enumerate(boundary_starts):
            end = boundary_starts[b_idx + 1] if b_idx + 1 < len(boundary_starts) else n
            concept = self.tagger.tag(sentences[start:end])

            if b_idx == 0:
                reason = "start"
            else:
                _, combined, ont_sig, emb_sig, _ = boundary_events[b_idx - 1]
                reason = (
                    f"hybrid: score={combined:.3f} "
                    f"[ontology={ont_sig:.2f} | embed_dissim={emb_sig:.2f}] "
                    f"→ '{concept.concept}' ({concept.concept_en})"
                )

            boundaries.append((start, reason, concept))

        return boundaries

    # ------------------------------------------------------------------
    # Boundary detection — keyword mode (original algorithm)
    # ------------------------------------------------------------------

    def _find_boundaries_keyword(
        self, sentences: List[str]
    ) -> List[tuple[int, str, ConceptResult]]:
        """
        Original keyword-based boundary detection using ConceptTagger.

        No models required — works fully offline on any machine.
        """
        n = len(sentences)
        boundaries: List[tuple[int, str, ConceptResult]] = []

        first_concept = _tag_window(self.tagger, sentences, 0, self.window_size)
        boundaries.append((0, "start", first_concept))

        prev_concept = first_concept
        in_chunk_since = 0

        i = self.step
        while i < n:
            curr_concept = _tag_window(self.tagger, sentences, i, self.window_size)
            chunk_len = i - in_chunk_since

            # Rule 1 — max size hard cap
            if chunk_len >= self.max_sentences:
                boundaries.append((
                    i,
                    f"max_size_limit ({self.max_sentences} sentences reached)",
                    curr_concept,
                ))
                prev_concept = curr_concept
                in_chunk_since = i

            # Rule 2 — ontological shift with sufficient confidence
            elif self.tagger.detect_shift(prev_concept, curr_concept, self.shift_threshold):
                boundaries.append((
                    i,
                    (
                        f"concept_shift: '{prev_concept.concept}' → "
                        f"'{curr_concept.concept}' "
                        f"(confidence={curr_concept.confidence:.2f})"
                    ),
                    curr_concept,
                ))
                prev_concept = curr_concept
                in_chunk_since = i

            # Rule 3 — intra-domain confidence drop (same concept, weakening signal)
            elif (
                prev_concept.concept == curr_concept.concept
                and chunk_len >= self.min_sentences
                and (prev_concept.confidence - curr_concept.confidence)
                    >= self.confidence_drop_threshold
            ):
                boundaries.append((
                    i,
                    (
                        f"confidence_drop: '{curr_concept.concept}' "
                        f"{prev_concept.confidence:.2f}→{curr_concept.confidence:.2f}"
                    ),
                    curr_concept,
                ))
                prev_concept = curr_concept
                in_chunk_since = i

            else:
                if curr_concept.confidence > prev_concept.confidence:
                    prev_concept = curr_concept

            i += self.step

        return boundaries

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _find_boundaries(
        self, sentences: List[str]
    ) -> List[tuple[int, str, ConceptResult]]:
        if self.use_ensemble:
            try:
                if self.ontology_weight > 0.0:
                    return self._find_boundaries_hybrid(sentences)
                return self._find_boundaries_ensemble(sentences)
            except Exception as exc:
                logger.warning("EnsembleEmbedder failed (%s), falling back to keyword mode.", exc)
        return self._find_boundaries_keyword(sentences)

    # ------------------------------------------------------------------
    # Chunk assembly from boundaries
    # ------------------------------------------------------------------

    def _assemble(
        self,
        sentences: List[str],
        boundaries: List[tuple[int, str, ConceptResult]],
    ) -> List[OntologyChunk]:
        """Convert boundary markers into OntologyChunk objects."""
        raw: List[OntologyChunk] = []
        for k, (start, reason, concept) in enumerate(boundaries):
            end = boundaries[k + 1][0] if k + 1 < len(boundaries) else len(sentences)
            raw.append(_build_chunk(sentences[start:end], concept, k, reason))
        return raw

    # ------------------------------------------------------------------
    # Size enforcement (merge small, split large)
    # ------------------------------------------------------------------

    def _merge_small(self, chunks: List[OntologyChunk]) -> List[OntologyChunk]:
        """
        Merge chunks smaller than min_sentences into their nearest neighbour.
        Preference: merge into the *previous* chunk (topic continuation).
        """
        if not chunks:
            return chunks

        merged: List[OntologyChunk] = [chunks[0]]
        for curr in chunks[1:]:
            prev = merged[-1]
            if curr.sentence_count < self.min_sentences:
                combined = prev.sentences + curr.sentences
                dominant = prev if prev.confidence >= curr.confidence else curr
                merged[-1] = _build_chunk(
                    combined,
                    ConceptResult(
                        concept=dominant.concept,
                        concept_en=dominant.concept_en,
                        confidence=dominant.confidence,
                        keywords=dominant.keywords,
                    ),
                    prev.chunk_index,
                    prev.boundary_reason + " [merged_small_chunk]",
                )
            else:
                merged.append(curr)

        return merged

    def _split_large(self, chunks: List[OntologyChunk]) -> List[OntologyChunk]:
        """
        Mechanically split chunks larger than max_sentences into
        max_sentences-sized pieces (plain sentence splitting, no re-tagging).
        """
        result: List[OntologyChunk] = []
        for chunk in chunks:
            if chunk.sentence_count <= self.max_sentences:
                result.append(chunk)
                continue
            sents = chunk.sentences
            for i in range(0, len(sents), self.max_sentences):
                piece = sents[i: i + self.max_sentences]
                result.append(_build_chunk(
                    piece,
                    ConceptResult(
                        concept=chunk.concept,
                        concept_en=chunk.concept_en,
                        confidence=chunk.confidence,
                        keywords=chunk.keywords,
                    ),
                    len(result),
                    chunk.boundary_reason + f" [auto_split part {i // self.max_sentences + 1}]",
                ))
        return result

    def _reindex(self, chunks: List[OntologyChunk]) -> List[OntologyChunk]:
        for i, c in enumerate(chunks):
            c.chunk_index = i
        return chunks

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk Arabic text and return a list of base Chunk objects.

        Implements the BaseChunker interface. For richer output (concept
        labels, boundary reasons, etc.) use ``chunk_rich()`` instead.
        """
        return [c.to_base_chunk() for c in self.chunk_rich(text)]

    def chunk_rich(self, text: str) -> List[OntologyChunk]:
        """
        Full pipeline — returns OntologyChunk objects with all metadata.

        Parameters
        ----------
        text:
            Raw Arabic text (a document, article, passage, etc.).

        Returns
        -------
        List[OntologyChunk]
            Each chunk carries: text, concept, concept_en, confidence,
            keywords, sentence_count, chunk_index, boundary_reason.
        """
        if self.normalize_input:
            text = normalize(text)

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Trivial case: too short to split meaningfully
        if len(sentences) <= self.min_sentences:
            concept = self.tagger.tag(sentences)
            return [_build_chunk(sentences, concept, 0, "single_chunk_below_min")]

        boundaries = self._find_boundaries(sentences)
        chunks = self._assemble(sentences, boundaries)
        chunks = self._merge_small(chunks)
        chunks = self._split_large(chunks)
        chunks = self._reindex(chunks)
        return chunks

    def chunk_file(self, filepath: str | Path) -> List[OntologyChunk]:
        """
        Read a UTF-8 ``.txt`` file and return ontological chunks.

        Parameters
        ----------
        filepath:
            Path to the input text file.
        """
        text = Path(filepath).read_text(encoding="utf-8")
        return self.chunk_rich(text)

    def chunk_dicts(self, text: str) -> List[Dict]:
        """Return chunks in the standard dict schema (shared with baselines)."""
        return [c.to_dict() for c in self.chunk_rich(text)]

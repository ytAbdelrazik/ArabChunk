"""Abstract base class for all chunkers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Represents a single text chunk with optional metadata."""

    text: str
    index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:60].replace('\n', ' ')
        return f"Chunk(index={self.index}, words={len(self)}, text='{preview}...')"


# Sentence-boundary splitter shared by all baseline chunkers
_SENT_RE = re.compile(r'[.!?؟،\n]+')


def _count_sentences(text: str) -> int:
    """Estimate sentence count by splitting on common delimiters."""
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return max(1, len(parts))


class BaseChunker(ABC):
    """Abstract base class defining the chunking interface."""

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into a list of Chunk objects.

        Args:
            text: Input text to chunk.

        Returns:
            Ordered list of Chunk objects.
        """

    # ------------------------------------------------------------------
    # Standard dict interface
    # ------------------------------------------------------------------

    def chunk_dicts(self, text: str) -> List[Dict]:
        """
        Return chunks as a list of standardised dicts.

        The dict schema is shared across ALL chunkers so that benchmarks
        can iterate over them uniformly:

        {
            "text":             str,
            "concept":          str | None,   # None for baselines
            "concept_en":       str | None,   # None for baselines
            "keywords":         List[str],    # [] for baselines
            "sentence_count":   int,
            "chunk_index":      int,
            "boundary_reason":  str,
        }

        Subclasses may override this for richer output (e.g. OntologyChunker).
        """
        return [
            {
                "text": c.text,
                "concept": c.metadata.get("concept"),
                "concept_en": c.metadata.get("concept_en"),
                "keywords": c.metadata.get("keywords", []),
                "sentence_count": c.metadata.get(
                    "sentence_count", _count_sentences(c.text)
                ),
                "chunk_index": c.index,
                "boundary_reason": c.metadata.get("boundary_reason", "baseline"),
            }
            for c in self.chunk(text)
        ]

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def chunk_batch(self, texts: List[str]) -> List[List[Chunk]]:
        """Chunk multiple texts, returning one list per input."""
        return [self.chunk(t) for t in texts]

    def chunk_dicts_batch(self, texts: List[str]) -> List[List[Dict]]:
        """chunk_dicts over multiple texts."""
        return [self.chunk_dicts(t) for t in texts]

    @staticmethod
    def reconstruct(chunks: List[Chunk], separator: str = ' ') -> str:
        """Reconstruct original text from a list of Chunk objects."""
        return separator.join(c.text for c in chunks)

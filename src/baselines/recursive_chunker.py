"""
Recursive character-based chunking baseline.

Tries to split on a hierarchy of separators (paragraph → sentence →
word) until every piece is at or below the target character count.
When a piece is already within the limit it is kept whole; when it
still exceeds the limit the next separator in the list is tried.
Overlap is applied by re-prepending the tail of the previous chunk.

This mirrors the behaviour of LangChain's RecursiveCharacterTextSplitter
adapted for Arabic punctuation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..chunker.base_chunker import BaseChunker, Chunk, _count_sentences


class RecursiveChunker(BaseChunker):
    """
    Hierarchical text splitter with paragraph → sentence → word priority.

    Parameters
    ----------
    chunk_size:
        Maximum chunk length in *characters*.
    overlap:
        Character overlap between consecutive chunks.
    separators:
        Ordered list of separator strings to try.  Default hierarchy is
        tuned for Modern Standard Arabic text.
    """

    DEFAULT_SEPARATORS: List[str] = [
        '\n\n',   # paragraph
        '\n',     # line break
        '؟',      # Arabic question mark
        '.',      # full stop
        '!',      # exclamation
        '،',      # Arabic comma
        ' ',      # word boundary
        '',       # character boundary (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators: List[str] = (
            separators if separators is not None else self.DEFAULT_SEPARATORS
        )

    # ------------------------------------------------------------------
    # Internal splitting logic
    # ------------------------------------------------------------------

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split *text* until every piece <= chunk_size chars.
        Returns a flat list of text fragments (no overlap applied yet).
        """
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ''
        next_seps = separators[1:] if len(separators) > 1 else ['']

        parts = text.split(sep) if sep else list(text)
        merged: List[str] = []
        current = ''

        for part in parts:
            glue = (current + sep + part).strip() if current else part.strip()
            if len(glue) <= self.chunk_size:
                current = glue
            else:
                if current:
                    merged.append(current)
                if len(part) > self.chunk_size:
                    # Part itself too big — recurse with the next separator
                    merged.extend(self._split_recursive(part, next_seps))
                    current = ''
                else:
                    current = part.strip()

        if current:
            merged.append(current)

        return merged

    def _apply_overlap(self, fragments: List[str]) -> List[str]:
        """
        Given a list of non-overlapping fragments, add character-level
        overlap so each chunk ends with up to *overlap* chars from the
        next chunk prepended.
        """
        if self.overlap == 0 or len(fragments) <= 1:
            return fragments

        result = [fragments[0]]
        for i in range(1, len(fragments)):
            # Take the tail of the *previous* fragment as prefix
            prefix = fragments[i - 1][-self.overlap:].strip()
            combined = (prefix + ' ' + fragments[i]).strip() if prefix else fragments[i]
            result.append(combined[:self.chunk_size])   # hard cap
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[Chunk]:
        """
        Recursively split text into size-bounded chunks.

        Returns
        -------
        List[Chunk]
        """
        raw = self._split_recursive(text.strip(), self.separators)
        with_overlap = self._apply_overlap(raw)

        chunks: List[Chunk] = []
        for i, frag in enumerate(with_overlap):
            frag = frag.strip()
            if not frag:
                continue
            chunks.append(Chunk(
                text=frag,
                index=len(chunks),
                metadata={
                    "concept": None,
                    "concept_en": None,
                    "keywords": [],
                    "sentence_count": _count_sentences(frag),
                    "boundary_reason": (
                        f"recursive_split(size={self.chunk_size},"
                        f"overlap={self.overlap})"
                    ),
                },
            ))
        return chunks

    def chunk_dicts(self, text: str) -> List[Dict]:
        """Return recursive chunks in the standard dict schema."""
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

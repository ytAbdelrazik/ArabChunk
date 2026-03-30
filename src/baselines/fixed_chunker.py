"""
Fixed token-size chunking baseline.

Splits Arabic text into windows of exactly N whitespace-delimited tokens
with an optional sliding overlap.  No linguistic analysis whatsoever —
serves as the simplest possible baseline.
"""

from __future__ import annotations

from typing import Dict, List

from ..chunker.base_chunker import BaseChunker, Chunk, _count_sentences


class FixedChunker(BaseChunker):
    """
    Splits text into fixed-size windows of N tokens with optional overlap.

    Parameters
    ----------
    chunk_size:
        Number of whitespace-delimited tokens per window.
    overlap:
        Number of tokens shared between consecutive windows.
        Must be strictly less than chunk_size.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into fixed-size token windows.

        Returns
        -------
        List[Chunk]
            chunk.metadata contains the standard dict fields so that
            BaseChunker.chunk_dicts() produces the correct schema.
        """
        tokens = text.split()
        if not tokens:
            return []

        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap
        token_pos = 0
        idx = 0

        while token_pos < len(tokens):
            window = tokens[token_pos: token_pos + self.chunk_size]
            chunk_text = ' '.join(window)
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                metadata={
                    "concept": None,
                    "concept_en": None,
                    "keywords": [],
                    "sentence_count": _count_sentences(chunk_text),
                    "boundary_reason": f"fixed_window(size={self.chunk_size},overlap={self.overlap})",
                },
            ))
            token_pos += step
            idx += 1

        return chunks

    def chunk_dicts(self, text: str) -> List[Dict]:
        """Return fixed chunks in the standard dict schema."""
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

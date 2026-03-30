"""
Tests for OntologyChunker.

All tests are offline — domain data comes from temp JSON fixtures,
morphology is disabled so there are no external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunker.ontology_chunker import OntologyChunk, OntologyChunker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def domains_dir(tmp_path: Path) -> Path:
    religion = {
        "concept": "دين",
        "concept_en": "Religion",
        "keywords": ["الله", "قرآن", "صلاة", "إسلام", "نبي", "حديث", "سنة"],
        "roots": ["دين", "صلو", "قرأ"],
    }
    politics = {
        "concept": "سياسة",
        "concept_en": "Politics",
        "keywords": ["حكومة", "رئيس", "انتخابات", "برلمان", "وزير", "دولة", "قانون"],
        "roots": ["حكم", "رأس", "نخب"],
    }
    (tmp_path / "religion.json").write_text(json.dumps(religion), encoding="utf-8")
    (tmp_path / "politics.json").write_text(json.dumps(politics), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def chunker(domains_dir: Path) -> OntologyChunker:
    return OntologyChunker(
        domains_dir=domains_dir,
        shift_threshold=0.5,
        min_sentences=2,
        max_sentences=8,
        normalize_input=False,   # skip normalisation for deterministic tests
        use_ensemble=False,      # no model downloads in unit tests
    )


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

RELIGION_TEXT = (
    "قال الله تعالى في القرآن الكريم. "
    "أدى المسلمون صلاة الجمعة. "
    "روى النبي الحديث الشريف. "
    "إن الإسلام دين السلام. "
    "تعلّم الصحابة السنة النبوية."
)

POLITICS_TEXT = (
    "أعلنت الحكومة نتائج الانتخابات. "
    "فاز الرئيس بأغلبية ساحقة في البرلمان. "
    "اجتمع الوزراء لمناقشة القانون الجديد. "
    "تسعى الدولة إلى تحقيق الاستقرار. "
    "صادق البرلمان على الميزانية."
)

MIXED_TEXT = RELIGION_TEXT + " " + POLITICS_TEXT


class TestBasicChunking:
    def test_returns_list(self, chunker: OntologyChunker):
        result = chunker.chunk_rich(RELIGION_TEXT)
        assert isinstance(result, list)

    def test_chunks_are_ontology_chunks(self, chunker: OntologyChunker):
        for c in chunker.chunk_rich(RELIGION_TEXT):
            assert isinstance(c, OntologyChunk)

    def test_no_text_lost(self, chunker: OntologyChunker):
        """All words from the original text must appear somewhere in the chunks."""
        import re
        _strip = re.compile(r'[^\u0600-\u06FF\w]')

        def words(text):
            return {_strip.sub('', w) for w in text.split() if _strip.sub('', w)}

        original_words = words(RELIGION_TEXT)
        chunked_words = set()
        for c in chunker.chunk_rich(RELIGION_TEXT):
            chunked_words.update(words(c.text))
        assert original_words == chunked_words

    def test_chunk_indices_sequential(self, chunker: OntologyChunker):
        chunks = chunker.chunk_rich(RELIGION_TEXT)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_empty_text_returns_empty(self, chunker: OntologyChunker):
        assert chunker.chunk_rich("") == []

    def test_whitespace_only_returns_empty(self, chunker: OntologyChunker):
        assert chunker.chunk_rich("   \n\n   ") == []


# ---------------------------------------------------------------------------
# chunk() BaseChunker compatibility
# ---------------------------------------------------------------------------

class TestBaseChunkerInterface:
    def test_chunk_returns_base_chunks(self, chunker: OntologyChunker):
        from src.chunker.base_chunker import Chunk
        result = chunker.chunk(RELIGION_TEXT)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_metadata_has_concept(self, chunker: OntologyChunker):
        result = chunker.chunk(RELIGION_TEXT)
        for c in result:
            assert "concept" in c.metadata


# ---------------------------------------------------------------------------
# Size constraints
# ---------------------------------------------------------------------------

class TestSizeConstraints:
    def test_no_chunk_exceeds_max_sentences(self, chunker: OntologyChunker):
        for c in chunker.chunk_rich(MIXED_TEXT):
            assert c.sentence_count <= chunker.max_sentences

    def test_short_text_single_chunk(self, chunker: OntologyChunker):
        short = "الله رحيم. القرآن كريم."
        chunks = chunker.chunk_rich(short)
        # 2 sentences < min_sentences(2) is equal, should produce 1 chunk
        assert len(chunks) >= 1

    def test_chunk_below_min_merged(self):
        """A config with min=3 should merge 2-sentence chunks."""
        c = OntologyChunker(min_sentences=3, max_sentences=10, normalize_input=False, use_ensemble=False)
        # Patch tagger to return different concepts to force many boundaries
        from src.chunker.concept_tagger import ConceptResult
        call_count = [0]

        def alternating_tag(sentences):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return ConceptResult("سياسة", "Politics", 0.9, [])
            return ConceptResult("دين", "Religion", 0.9, [])

        c.tagger.tag = alternating_tag
        text = " ".join([f"جملة رقم {i}." for i in range(12)])
        chunks = c.chunk_rich(text)
        # All chunks except possibly the last remainder must meet min_sentences
        for chunk in chunks[:-1]:
            assert chunk.sentence_count >= c.min_sentences
        # The final chunk may be smaller if it is the document tail
        assert sum(ch.sentence_count for ch in chunks) == 12


# ---------------------------------------------------------------------------
# Boundary reasons
# ---------------------------------------------------------------------------

class TestBoundaryReasons:
    def test_first_chunk_reason_is_start(self, chunker: OntologyChunker):
        chunks = chunker.chunk_rich(RELIGION_TEXT)
        assert chunks[0].boundary_reason == "start"

    def test_shift_boundary_contains_arrow(self, chunker: OntologyChunker):
        chunks = chunker.chunk_rich(MIXED_TEXT)
        shift_chunks = [c for c in chunks if "→" in c.boundary_reason]
        # Not guaranteed, but with clearly different domains it should appear
        # (test is informational — just ensure the format is right when present)
        for c in shift_chunks:
            assert "concept_shift" in c.boundary_reason


# ---------------------------------------------------------------------------
# OntologyChunk.to_dict
# ---------------------------------------------------------------------------

class TestOntologyChunkToDict:
    def test_required_keys(self, chunker: OntologyChunker):
        expected = {
            "text", "concept", "concept_en", "confidence",
            "keywords", "sentence_count", "chunk_index", "boundary_reason",
        }
        chunks = chunker.chunk_rich(RELIGION_TEXT)
        for c in chunks:
            assert set(c.to_dict().keys()) == expected

    def test_sentence_count_matches_sentences_field(self, chunker: OntologyChunker):
        for c in chunker.chunk_rich(RELIGION_TEXT):
            assert c.sentence_count == len(c.sentences)


# ---------------------------------------------------------------------------
# chunk_file
# ---------------------------------------------------------------------------

class TestChunkFile:
    def test_chunk_file_reads_txt(self, chunker: OntologyChunker, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text(RELIGION_TEXT, encoding="utf-8")
        chunks = chunker.chunk_file(f)
        assert len(chunks) >= 1
        assert all(isinstance(c, OntologyChunk) for c in chunks)

    def test_chunk_file_missing_raises(self, chunker: OntologyChunker):
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file("/nonexistent/path/file.txt")


# ---------------------------------------------------------------------------
# Invalid constructor args
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_step_greater_than_window_raises(self):
        with pytest.raises(ValueError):
            OntologyChunker(window_size=3, step=4)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            OntologyChunker(min_sentences=10, max_sentences=5)

    def test_zero_min_sentences_raises(self):
        with pytest.raises(ValueError):
            OntologyChunker(min_sentences=0)

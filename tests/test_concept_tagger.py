"""
Tests for ConceptTagger.

All tests are offline — no network, no CAMeL DB required.
The morphology layer is mocked so the tagger falls back to heuristic
stemming, which is deterministic and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chunker.concept_tagger import ConceptResult, ConceptTagger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_domains(tmp_path: Path) -> Path:
    """Create a minimal set of domain JSON files for testing."""
    religion = {
        "concept": "دين",
        "concept_en": "Religion",
        "keywords": ["الله", "قرآن", "صلاة", "إسلام", "نبي"],
        "roots": ["دين", "صلو", "قرأ", "سلم"],
    }
    politics = {
        "concept": "سياسة",
        "concept_en": "Politics",
        "keywords": ["حكومة", "رئيس", "انتخابات", "برلمان", "وزير"],
        "roots": ["حكم", "رأس", "نخب", "وزر"],
    }
    (tmp_path / "religion.json").write_text(json.dumps(religion), encoding="utf-8")
    (tmp_path / "politics.json").write_text(json.dumps(politics), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def tagger(temp_domains: Path) -> ConceptTagger:
    """Return a ConceptTagger with test domains and no CAMeL dependency."""
    return ConceptTagger(domains_dir=temp_domains, use_morphology=False)


# ---------------------------------------------------------------------------
# Domain loading
# ---------------------------------------------------------------------------

class TestDomainLoading:
    def test_loads_all_json_files(self, tagger: ConceptTagger):
        assert len(tagger._domains) == 2

    def test_domain_names(self, tagger: ConceptTagger):
        names = tagger.domain_names
        assert "دين" in names
        assert "سياسة" in names

    def test_empty_dir_loads_nothing(self, tmp_path: Path):
        t = ConceptTagger(domains_dir=tmp_path, use_morphology=False)
        assert t._domains == []

    def test_malformed_json_is_skipped(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
        t = ConceptTagger(domains_dir=tmp_path, use_morphology=False)
        assert t._domains == []

    def test_add_domain_at_runtime(self, tagger: ConceptTagger, tmp_path: Path):
        science = {
            "concept": "علم",
            "concept_en": "Science",
            "keywords": ["بحث", "تجربة"],
            "roots": ["بحث"],
        }
        p = tmp_path / "science.json"
        p.write_text(json.dumps(science), encoding="utf-8")
        tagger.add_domain(p)
        assert "علم" in tagger.domain_names


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

class TestTagging:
    def test_religion_text_tagged_correctly(self, tagger: ConceptTagger):
        sentences = [
            "قرأ المسلمون القرآن الكريم.",
            "أدى المصلون صلاة الفجر في المسجد.",
            "قال النبي صلى الله عليه وسلم.",
        ]
        result = tagger.tag(sentences)
        assert result.concept == "دين"
        assert result.concept_en == "Religion"
        assert result.confidence > 0.0

    def test_politics_text_tagged_correctly(self, tagger: ConceptTagger):
        sentences = [
            "أعلنت الحكومة عن نتائج الانتخابات.",
            "فاز الرئيس بأغلبية ساحقة.",
            "يجتمع البرلمان غداً.",
        ]
        result = tagger.tag(sentences)
        assert result.concept == "سياسة"
        assert result.confidence > 0.0

    def test_unknown_text_returns_general(self, tagger: ConceptTagger):
        sentences = ["كلمة", "أخرى", "غريبة"]
        result = tagger.tag(sentences)
        assert result.concept == "عام"
        assert result.confidence == 0.0
        assert result.keywords == []

    def test_returns_concept_result_instance(self, tagger: ConceptTagger):
        result = tagger.tag(["الله", "قرآن", "صلاة"])
        assert isinstance(result, ConceptResult)

    def test_confidence_bounded(self, tagger: ConceptTagger):
        result = tagger.tag(["الله", "قرآن", "صلاة", "إسلام", "نبي"])
        assert 0.0 <= result.confidence <= 1.0

    def test_keywords_subset_of_input(self, tagger: ConceptTagger):
        sentences = ["الله أكبر", "القرآن الكريم", "صلاة الجمعة"]
        result = tagger.tag(sentences)
        all_words = set(" ".join(sentences).split())
        for kw in result.keywords:
            assert kw in all_words  # keywords come from raw tokens, not stems


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_same_input_returns_cached_result(self, tagger: ConceptTagger):
        sentences = ["الله", "قرآن", "صلاة"]
        r1 = tagger.tag(sentences)
        # Poison the domain data to ensure second call uses cache
        tagger._domains = []
        r2 = tagger.tag(sentences)
        assert r1 is r2  # Same object from cache

    def test_different_input_not_cached(self, tagger: ConceptTagger):
        r1 = tagger.tag(["الله", "قرآن"])
        r2 = tagger.tag(["حكومة", "رئيس"])
        assert r1.concept != r2.concept

    def test_clear_cache(self, tagger: ConceptTagger):
        sentences = ["الله", "قرآن"]
        tagger.tag(sentences)
        assert len(tagger._cache) == 1
        tagger.clear_cache()
        assert len(tagger._cache) == 0

    def test_cache_keyed_on_sha256(self, tagger: ConceptTagger):
        sentences = ["الله", "قرآن"]
        tagger.tag(sentences)
        expected_key = hashlib.sha256(" ".join(sentences).encode("utf-8")).hexdigest()
        assert expected_key in tagger._cache


# ---------------------------------------------------------------------------
# detect_shift
# ---------------------------------------------------------------------------

class TestDetectShift:
    def _make(self, concept: str, concept_en: str, confidence: float) -> ConceptResult:
        return ConceptResult(concept=concept, concept_en=concept_en, confidence=confidence)

    def test_same_concept_no_shift(self, tagger: ConceptTagger):
        a = self._make("دين", "Religion", 0.9)
        b = self._make("دين", "Religion", 0.9)
        assert tagger.detect_shift(a, b) is False

    def test_different_concept_high_confidence_is_shift(self, tagger: ConceptTagger):
        a = self._make("دين", "Religion", 0.9)
        b = self._make("سياسة", "Politics", 0.8)
        assert tagger.detect_shift(a, b, threshold=0.7) is True

    def test_different_concept_low_confidence_not_shift(self, tagger: ConceptTagger):
        a = self._make("دين", "Religion", 0.9)
        b = self._make("سياسة", "Politics", 0.4)
        assert tagger.detect_shift(a, b, threshold=0.7) is False

    def test_threshold_boundary_exact(self, tagger: ConceptTagger):
        a = self._make("دين", "Religion", 0.9)
        b = self._make("سياسة", "Politics", 0.7)
        assert tagger.detect_shift(a, b, threshold=0.7) is True

    def test_threshold_just_below(self, tagger: ConceptTagger):
        a = self._make("دين", "Religion", 0.9)
        b = self._make("سياسة", "Politics", 0.699)
        assert tagger.detect_shift(a, b, threshold=0.7) is False

    def test_general_to_specific_with_confidence(self, tagger: ConceptTagger):
        a = self._make("عام", "General", 0.0)
        b = self._make("علم", "Science", 0.8)
        assert tagger.detect_shift(a, b, threshold=0.7) is True


# ---------------------------------------------------------------------------
# ConceptResult.to_dict
# ---------------------------------------------------------------------------

class TestConceptResultToDict:
    def test_keys_present(self):
        r = ConceptResult(concept="دين", concept_en="Religion", confidence=0.85, keywords=["قرآن"])
        d = r.to_dict()
        assert set(d.keys()) == {"concept", "concept_en", "confidence", "keywords"}

    def test_values_correct(self):
        r = ConceptResult(concept="علم", concept_en="Science", confidence=0.6, keywords=["بحث"])
        d = r.to_dict()
        assert d["concept"] == "علم"
        assert d["confidence"] == 0.6
        assert d["keywords"] == ["بحث"]

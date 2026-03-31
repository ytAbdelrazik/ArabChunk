"""
Offline ontological concept tagger for Arabic text windows.

No API keys, no network calls. Concept detection is driven by:
  1. Domain JSON files in data/domains/ (add a JSON → add a domain)
  2. Arabic morphological root matching via CAMeL Tools (with heuristic fallback)
  3. TF-normalised keyword overlap scoring

Every result is cached in-process by the SHA-256 of the input text,
so repeated calls on identical windows are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_CACHE_SIZE = 10_000   # max in-process cache entries before FIFO eviction

# ---------------------------------------------------------------------------
# Optional CAMeL Tools integration (graceful degradation if not installed)
# ---------------------------------------------------------------------------
try:
    from ..preprocessing.morphology import MorphologyAnalyzer as _MorphAnalyzer
    _CAMEL_AVAILABLE = True
except Exception:
    _CAMEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Keyword normalization at domain load time
# ---------------------------------------------------------------------------
try:
    from ..preprocessing.normalizer import normalize as _normalize_kw
except Exception:
    def _normalize_kw(text: str, **kwargs) -> str:  # type: ignore[misc]
        return text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConceptResult:
    """Holds the tagging result for a single text window."""

    concept: str          # Primary Arabic concept label (e.g. "دين")
    concept_en: str       # English gloss (e.g. "Religion")
    confidence: float     # Normalised score in [0.0, 1.0]
    keywords: List[str] = field(default_factory=list)   # Matched Arabic terms

    def to_dict(self) -> Dict:
        return {
            "concept": self.concept,
            "concept_en": self.concept_en,
            "confidence": self.confidence,
            "keywords": self.keywords,
        }


# ---------------------------------------------------------------------------
# Lightweight Arabic root heuristic (fallback when CAMeL is unavailable)
# ---------------------------------------------------------------------------

_ARABIC_PREFIXES = re.compile(
    r'^(وال|فال|بال|كال|لل|وب|فب|كب|وك|لب|ال|و|ف|ب|ك|ل|س|)'
)
_ARABIC_SUFFIXES = re.compile(
    r'(ون|ين|ات|تين|تان|ان|ة|ه|ي|ك|نا|كم|هم|هن|تم|تن|وا|ا|ن|ت|)$'
)


def _heuristic_stem(word: str) -> str:
    """Strip common Arabic prefixes/suffixes to approximate a stem."""
    original = word
    word = _ARABIC_PREFIXES.sub('', word)
    word = _ARABIC_SUFFIXES.sub('', word)
    # Require at least 3 chars to protect trilateral roots; fall back to original
    return word if len(word) >= 3 else original


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ConceptTagger:
    """
    Tags a window of 3-5 Arabic sentences with an ontological concept.

    Designed to be fully offline and reproducible. The domain vocabulary
    lives in JSON files under ``data/domains/``; adding a new JSON file
    instantly makes a new concept detectable with no code changes.

    Parameters
    ----------
    domains_dir:
        Path to the directory containing domain ``*.json`` files.
        Defaults to ``<repo_root>/data/domains/``.
    use_morphology:
        Whether to use CAMeL Tools for root extraction when available.
        If False (or if CAMeL is not installed), a heuristic stemmer is
        used instead.
    """

    # Resolved at import time relative to this file's location
    _DEFAULT_DOMAINS = Path(__file__).resolve().parent.parent.parent / "data" / "domains"

    def __init__(
        self,
        domains_dir: Optional[Path] = None,
        use_morphology: bool = True,
    ) -> None:
        self._domains_dir = Path(domains_dir) if domains_dir else self._DEFAULT_DOMAINS
        self._domains: List[Dict] = []
        self._cache: Dict[str, ConceptResult] = {}
        self._morph = None

        if use_morphology and _CAMEL_AVAILABLE:
            try:
                self._morph = _MorphAnalyzer()
            except Exception:
                pass  # CAMeL DB not downloaded; fall back to heuristic

        self._load_domains()

    # ------------------------------------------------------------------
    # Domain loading
    # ------------------------------------------------------------------

    def _load_domains(self) -> None:
        """Load all JSON domain files from the domains directory."""
        if not self._domains_dir.exists():
            return
        for path in sorted(self._domains_dir.glob("*.json")):
            try:
                with path.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                # Normalize keywords so they match the normalized input text
                data["_kw_set"] = {_normalize_kw(k) for k in data.get("keywords", [])}
                data["_root_set"] = {_normalize_kw(r) for r in data.get("roots", [])}
                self._domains.append(data)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed domain file: %s", path)
                continue

    def add_domain(self, path: Path) -> None:
        """
        Dynamically add a domain from a JSON file at runtime.

        The JSON must contain at least ``concept``, ``concept_en``,
        and one of ``keywords`` / ``roots``.
        """
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        data["_kw_set"] = {_normalize_kw(k) for k in data.get("keywords", [])}
        data["_root_set"] = {_normalize_kw(r) for r in data.get("roots", [])}
        self._domains.append(data)

    # ------------------------------------------------------------------
    # Token / root extraction
    # ------------------------------------------------------------------

    def _tokens_and_stems(self, text: str) -> List[str]:
        """
        Return a combined list of raw tokens and their roots/stems.

        This union strategy maximises recall: a keyword match OR a root
        match counts as a hit.
        """
        tokens = text.split()
        stems: List[str] = []

        if self._morph:
            try:
                roots = self._morph.extract_roots(tokens)
                stems = [r for r in roots if r]
            except Exception:
                stems = [_heuristic_stem(t) for t in tokens]
        else:
            stems = [_heuristic_stem(t) for t in tokens]

        return tokens + stems   # duplicates are fine; set() used at match time

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_domain(
        self, all_tokens: List[str], domain: Dict
    ) -> Tuple[float, List[str]]:
        """
        Compute a TF-normalised overlap score for one domain.

        Score = |matched_unique_terms| / sqrt(|domain_vocabulary|)

        This rewards specificity (small, tight domains score higher than
        massive catch-all ones for the same number of hits).
        """
        vocab = domain["_kw_set"] | domain["_root_set"]
        if not vocab:
            return 0.0, []

        matched = {t for t in all_tokens if t in vocab}
        if not matched:
            return 0.0, []

        score = len(matched) / math.sqrt(len(vocab))
        # Return original keywords (not stems) that matched for readability
        readable = [t for t in all_tokens if t in domain["_kw_set"]]
        return score, list(dict.fromkeys(readable))  # deduplicate, preserve order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, sentences: List[str]) -> ConceptResult:
        """
        Tag a window of 3-5 Arabic sentences with an ontological concept.

        Results are memoised by the SHA-256 of the joined text, so
        repeated calls on identical windows cost nothing.

        Parameters
        ----------
        sentences:
            A list of 3-5 Arabic sentence strings.

        Returns
        -------
        ConceptResult
            Populated with concept, concept_en, confidence, and keywords.
            Returns a "عام / General" result when no domain matches.
        """
        text = " ".join(sentences)
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        all_tokens = self._tokens_and_stems(text)

        best_score = 0.0
        best_domain: Optional[Dict] = None
        best_keywords: List[str] = []

        for domain in self._domains:
            score, matched = self._score_domain(all_tokens, domain)
            if score > best_score:
                best_score = score
                best_domain = domain
                best_keywords = matched

        if best_domain is None or best_score == 0.0:
            result = ConceptResult(
                concept="عام",
                concept_en="General",
                confidence=0.0,
                keywords=[],
            )
        else:
            # Map raw score to [0, 1] via a soft cap at score=0.5
            # (a score of 0.5 or above is considered very confident)
            confidence = round(min(1.0, best_score / 0.5), 3)
            result = ConceptResult(
                concept=best_domain["concept"],
                concept_en=best_domain["concept_en"],
                confidence=confidence,
                keywords=best_keywords[:10],
            )

        if len(self._cache) >= _MAX_CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))  # evict oldest (FIFO)
        self._cache[cache_key] = result
        return result

    def detect_shift(
        self,
        concept_a: ConceptResult,
        concept_b: ConceptResult,
        threshold: float = 0.7,
    ) -> bool:
        """
        Return True when an ontological topic shift is detected.

        A shift is declared when ALL of the following hold:
          - The concept labels differ (string inequality).
          - The *incoming* concept ``concept_b`` has confidence >= threshold
            (i.e. the new topic is strongly supported by evidence).

        Parameters
        ----------
        concept_a:
            ConceptResult for the previous sliding window.
        concept_b:
            ConceptResult for the current sliding window.
        threshold:
            Minimum confidence for ``concept_b`` to accept the shift.

        Returns
        -------
        bool
        """
        if concept_a.concept == concept_b.concept:
            return False
        return concept_b.confidence >= threshold

    def clear_cache(self) -> None:
        """Evict all cached tagging results."""
        self._cache.clear()

    @property
    def domain_names(self) -> List[str]:
        """Return the concept labels of all loaded domains."""
        return [d["concept"] for d in self._domains]

"""Arabic morphological analysis: root extraction and lemmatization."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False


class MorphologyAnalyzer:
    """Wrapper around CAMeL Tools morphological analyzer."""

    def __init__(self, db_name: str = 'calima-msa-r13'):
        """
        Initialize the morphology analyzer.

        Args:
            db_name: CAMeL Tools morphology database name.
        """
        self._analyzer: Optional[object] = None
        self._db_name = db_name

    def _load(self) -> None:
        if not CAMEL_AVAILABLE:
            raise ImportError(
                "camel-tools is required for morphological analysis. "
                "Install with: pip install camel-tools"
            )
        if self._analyzer is None:
            db = MorphologyDB.builtin_db(self._db_name)
            self._analyzer = Analyzer(db)

    def analyze(self, word: str) -> List[Dict]:
        """
        Return all morphological analyses for a word.

        Args:
            word: Single Arabic word token.

        Returns:
            List of analysis dicts from CAMeL Tools.
        """
        self._load()
        return self._analyzer.analyze(word)

    def get_lemma(self, word: str) -> str:
        """
        Return the most likely lemma for a word.

        Args:
            word: Single Arabic word token.

        Returns:
            Lemma string, or the original word if unavailable.
        """
        analyses = self.analyze(word)
        if analyses:
            return analyses[0].get('lex', word)
        return word

    def get_root(self, word: str) -> Optional[str]:
        """
        Return the trilateral/quadrilateral root for a word.

        Args:
            word: Single Arabic word token.

        Returns:
            Root string (e.g. 'كتب') or None if not found.
        """
        analyses = self.analyze(word)
        for analysis in analyses:
            root = analysis.get('root')
            if root and root != 'NOAN':
                return root
        return None

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize a list of tokens.

        Args:
            tokens: List of Arabic word tokens.

        Returns:
            List of lemmas corresponding to each token.
        """
        return [self.get_lemma(t) for t in tokens]

    def extract_roots(self, tokens: List[str]) -> List[Optional[str]]:
        """
        Extract roots for a list of tokens.

        Args:
            tokens: List of Arabic word tokens.

        Returns:
            List of root strings (None where unavailable).
        """
        return [self.get_root(t) for t in tokens]

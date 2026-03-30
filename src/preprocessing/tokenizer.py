"""CAMeL Tools tokenizer wrapper for Arabic text."""

from __future__ import annotations

from typing import List

try:
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.utils.dediac import dediac_ar
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False


def tokenize(text: str) -> List[str]:
    """
    Tokenize Arabic text into word tokens.

    Uses CAMeL Tools simple word tokenizer if available,
    falls back to whitespace splitting otherwise.

    Args:
        text: Preprocessed Arabic text.

    Returns:
        List of word tokens.
    """
    if CAMEL_AVAILABLE:
        return simple_word_tokenize(text)
    # Fallback: whitespace tokenization
    return text.split()


def sentence_tokenize(text: str, delimiters: str = '.!?؟،\n') -> List[str]:
    """
    Split Arabic text into sentences.

    Args:
        text: Arabic text.
        delimiters: Characters used as sentence boundaries.

    Returns:
        List of sentence strings.
    """
    import re
    pattern = '[' + ''.join(re.escape(ch) for ch in delimiters) + ']+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def dediacritize(text: str) -> str:
    """
    Remove diacritics using CAMeL Tools if available.

    Args:
        text: Arabic text with diacritics.

    Returns:
        Text with diacritics removed.
    """
    if CAMEL_AVAILABLE:
        return dediac_ar(text)
    import re
    return re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)

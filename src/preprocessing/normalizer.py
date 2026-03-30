"""Arabic text normalization utilities."""

import re
import unicodedata


# Arabic Unicode ranges and characters
ARABIC_DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670]')
TATWEEL = re.compile(r'\u0640')
PUNCTUATION = re.compile(r'[^\w\s\u0600-\u06FF]')

# Normalization maps
ALEF_VARIANTS = str.maketrans('إأآٱ', 'اااا')
YEH_VARIANTS = str.maketrans('ى', 'ي')
HEH_VARIANTS = str.maketrans('ة', 'ه')


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel)."""
    return ARABIC_DIACRITICS.sub('', text)


def remove_tatweel(text: str) -> str:
    """Remove Arabic tatweel (kashida) character."""
    return TATWEEL.sub('', text)


def normalize_alef(text: str) -> str:
    """Normalize all alef variants to bare alef."""
    return text.translate(ALEF_VARIANTS)


def normalize_yeh(text: str) -> str:
    """Normalize alef maqsura to yeh."""
    return text.translate(YEH_VARIANTS)


def normalize_teh_marbuta(text: str) -> str:
    """Normalize teh marbuta to heh."""
    return text.translate(HEH_VARIANTS)


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization."""
    return unicodedata.normalize('NFC', text)


def normalize(
    text: str,
    remove_diacritics_flag: bool = True,
    remove_tatweel_flag: bool = True,
    normalize_alef_flag: bool = True,
    normalize_yeh_flag: bool = True,
    normalize_teh_marbuta_flag: bool = False,
) -> str:
    """
    Full Arabic text normalization pipeline.

    Args:
        text: Raw Arabic text.
        remove_diacritics_flag: Strip tashkeel marks.
        remove_tatweel_flag: Remove kashida characters.
        normalize_alef_flag: Unify alef variants.
        normalize_yeh_flag: Unify yeh/alef-maqsura.
        normalize_teh_marbuta_flag: Convert teh marbuta to heh.

    Returns:
        Normalized Arabic text.
    """
    text = normalize_unicode(text)
    if remove_diacritics_flag:
        text = remove_diacritics(text)
    if remove_tatweel_flag:
        text = remove_tatweel(text)
    if normalize_alef_flag:
        text = normalize_alef(text)
    if normalize_yeh_flag:
        text = normalize_yeh(text)
    if normalize_teh_marbuta_flag:
        text = normalize_teh_marbuta(text)
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

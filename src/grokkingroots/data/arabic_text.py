"""Minimal Arabic text utilities — self-contained, no external Arabic NLP deps."""
from __future__ import annotations

# The eight core diacritics + dagger alef.
DIACRITICS = set("ًٌٍَُِّْٰ")
TATWEEL = "ـ"


def strip_diacritics(text: str) -> str:
    """Remove diacritic marks, leaving the bare consonant/letter skeleton."""
    return "".join(c for c in str(text) if c not in DIACRITICS)


def remove_tatweel(text: str) -> str:
    return str(text).replace(TATWEEL, "")


def normalize_surface(text: str) -> str:
    """Primary-regime surface form: strip diacritics + tatweel, keep alif/hamza.

    Alif/hamza/ya distinctions are part of the morphological difficulty, so they
    are deliberately preserved here (an alif-normalized variant is a separate
    ablation, not the default).
    """
    return remove_tatweel(strip_diacritics(text)).strip()


# Lenient و/ي scoring: two roots count as the same
# answer if they differ only by waw<->ya in a radical position. Nothing else is
# relaxed (alif, alif-maqsura and the hamza carriers stay distinct).
_WAW_YA = str.maketrans("ي", "و")


def lenient_key(root: str) -> str:
    """Canonical form of a root under the و/ي-lenient criterion."""
    return str(root).translate(_WAW_YA)


def lenient_match(pred: str, gold: str) -> bool:
    return lenient_key(pred) == lenient_key(gold)

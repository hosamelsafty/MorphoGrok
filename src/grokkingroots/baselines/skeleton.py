"""Deterministic 'consonant-skeleton' baseline.

For sound triliteral roots the radicals are roughly the surface consonants once
long vowels / weak letters / clitics are removed. This baseline measures how much
of root extraction is trivial 'copy the consonants' — the floor a grokked model
must beat, especially on weak/hamzated roots where letters must be *restored*,
not copied, and where affixal consonants (e.g. the م of مكتوب) must be dropped.
"""
from __future__ import annotations

# Long vowels, weak letters, hamza carriers, ta-marbuta, tatweel — rarely radicals.
NONROOT = set("اوىيءأإؤئآةـ")
ARTICLE = "ال"


def skeleton_root(surface: str, n: int = 3) -> str:
    """Predict an n-radical root as the last n surface consonants after stripping
    the article and non-root letters. (Last-n beats first-n because most affixes
    that survive stripping are prefixes, e.g. م/ت/ن/مست.)"""
    s = surface[2:] if surface.startswith(ARTICLE) and len(surface) > 4 else surface
    cons = [c for c in s if c not in NONROOT]
    return "".join(cons[-n:]) if len(cons) >= n else "".join(cons)

"""Classify an Arabic root by type and assign a difficulty level L1-L4.

Root format follows Qabas: space-separated radicals, e.g. "ك ت ب", "د ح ر ج".
The difficulty ladder is the grokking 'cleanliness' axis (cleaner = more likely
to grok):
  L1  sound triliteral            (the clean, most 'algorithmic' regime)
  L2  weak or hamzated triliteral (assimilated / hollow / defective / hamzated)
  L3  doubled (geminate) triliteral
  L4  quadriliteral (or longer)
"""
from __future__ import annotations

WEAK = set("اوىيآ")     # alif, waw, alif-maqsura, ya, alif-madda
HAMZA = set("ءأإؤئآ")   # hamza carriers


def classify_root_type(root: str) -> dict:
    radicals = str(root).split()
    n = len(radicals)
    hamzated = any(ch in HAMZA for r in radicals for ch in r)
    info = {"n_radicals": n, "hamzated": hamzated}

    if n < 3 or n > 4:
        info.update(root_type="other", difficulty="L4" if n > 4 else "other")
        return info
    if n == 4:
        info.update(root_type="quadriliteral", difficulty="L4")
        return info

    r1, r2, r3 = radicals
    doubled = r2 == r3
    assimilated = r1 in WEAK      # mithāl
    hollow = r2 in WEAK          # ajwaf
    defective = r3 in WEAK        # nāqiṣ

    if doubled:
        root_type, difficulty = "doubled", "L3"
    elif assimilated or hollow or defective:
        root_type = "assimilated" if assimilated else "hollow" if hollow else "defective"
        difficulty = "L2"
    elif hamzated:
        root_type, difficulty = "hamzated", "L2"
    else:
        root_type, difficulty = "sound", "L1"
    info.update(root_type=root_type, difficulty=difficulty)
    return info

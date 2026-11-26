"""How many L2 test roots have a root in the inventory that differs only by w<->y.

Backs the sentence in the paper's section 5 (lenient scoring):
"85 of the 192 test roots have such a counterpart in the root inventory".

    python3 scripts/analysis/count_wy_counterparts.py
"""
import json
from collections import defaultdict

import pandas as pd

from grokkingroots.data.arabic_text import lenient_key

split = json.load(open("artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json"))
test_roots = sorted({y for _, y in split["test"]})
inventory = set(pd.read_parquet("artefacts/phase0/roots_dataset_v2clean.parquet")["root_clean"].astype(str))
classes = defaultdict(set)
for r in inventory:
    classes[lenient_key(r)].add(r)
n = sum(len(classes[lenient_key(r)]) > 1 for r in test_roots)
print(f"inventory {len(inventory)} roots; test roots {len(test_roots)}; "
      f"with a w/y-only counterpart: {n}")

"""Gen-2b: three targeted attacks on the deleted-radical case.

  k3_reveal  supports SELECTED to visibly contain the weak radical (vs gen-2's
             random sampling) -> every hidden query gets a witness. Measures
             the pure witness-availability effect; expected to lift the
             deleted-radical bucket toward the visible-case rate.
  kmix       training examples use k sampled from {1,2,3}; TEST is single-form
             (k=1). The deployable question gen-3 failed at: does mixed-k
             training transfer paradigm skill to the single-form interface?

Same query/test roots as gen-2 (fixed across conditions). DELIM-joined inputs,
zero model change.
"""
import hashlib, json, os, random, re
from pathlib import Path
import pandas as pd

DELIM = "؞"; SEED = 42
VARIANT = os.environ.get("GEN2_VARIANT", "v2clean")
_SUF = "" if VARIANT == "enrclean" else f"_{VARIANT}"
DATASET = f"artefacts/phase0/roots_dataset_{VARIANT}.parquet"
BASE = f"artefacts/phase0/splits_gen2{_SUF}/L2_p30_k1.json"
OUT = Path(f"artefacts/phase0/splits_gen2{_SUF}")
HAMZA = {"ء":"ءأإؤئآ","أ":"ءأإؤئآ","إ":"ءأإؤئآ","ؤ":"ءأإؤئآ","ئ":"ءأإؤئآ","آ":"ءأإؤئآ"}

def weak(root, rt):
    r = root.split()
    if rt == "hollow": return r[1]
    if rt == "defective": return r[2]
    if rt == "assimilated": return r[0]
    if rt == "hamzated":
        for x in r:
            if x in HAMZA: return x
    return None

def shows(w, s):
    if w is None: return True
    if w in HAMZA: return any(c in s for c in HAMZA[w])
    if w == "ي": return "ي" in s or "ى" in s
    return w in s

def sha(keys):
    return hashlib.sha256(json.dumps(sorted("\t".join(k) for k in keys),
                          ensure_ascii=False).encode()).hexdigest()

rng = random.Random(SEED)
df = pd.read_parquet(DATASET)
forms = df.groupby("root_clean")["surface_clean"].apply(lambda x: list(dict.fromkeys(x))).to_dict()
winfo = {}
for _, r in df.iterrows():
    winfo.setdefault(r.root_clean, weak(r.root, r.root_type))
base = json.loads(Path(BASE).read_text(encoding="utf-8"))
summary = json.loads((OUT / "_summary.json").read_text())

def supports_reveal(s, r, n):
    w = winfo.get(r)
    pool = [f for f in forms[r] if f != s]
    revealing = [f for f in pool if shows(w, f)]
    hiding = [f for f in pool if not shows(w, f)]
    rng.shuffle(revealing); rng.shuffle(hiding)
    out = (revealing + hiding)[:n]
    while len(out) < n: out.append(out[-1] if out else s)
    return out

def build(name, mode):
    cond = {}
    for part in ("train", "val", "test"):
        new = []
        for i, (s, r) in enumerate(base[part]):
            if mode == "reveal":
                k = 3
            else:  # kmix: train mixed k, val/test single-form
                k = rng.choice([1, 2, 3]) if part == "train" else 1
            sup = supports_reveal(s, r, k - 1) if k > 1 else []
            new.append([DELIM.join([s] + sup), r])
        cond[part] = new
    assert not ({p[0].split(DELIM)[0] for p in cond["train"]} &
                {p[0].split(DELIM)[0] for p in cond["test"]})
    meta = {"family": "gen2b", "mode": mode, "delim": DELIM, "base": BASE, "seed": SEED}
    payload = {"meta": meta, "test_sha256": sha(cond["test"]), **cond}
    (OUT / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    summary[name] = {**meta, "sizes": {p: len(v) for p, v in cond.items()},
                     "test_sha256": payload["test_sha256"][:12]}
    print(f"{name}: " + " ".join(f"{p}={len(v)}" for p, v in cond.items()))
    print(f"   ex: {cond['test'][1][0]!r} -> {cond['test'][1][1]}")

build("L2_p30_k3_reveal", "reveal")
build("L2_p30_kmix", "kmix")
(OUT / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("freeze updated")

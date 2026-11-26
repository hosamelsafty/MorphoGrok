# Figure scripts

```bash
pip install -e ".[figures]"
python scripts/figures/make_all.py                  # every figure, into figures/
python scripts/figures/make_all.py --only fig_selective
```

Run from the repository root, after the runs exist under `artefacts/runs/`.

- `data.py` loads everything the figures show. It uses only the standard
  library, so it can be checked without the plotting stack. Trajectories come
  from the run directories; headline accuracies are the values re-scored on the
  complete test set.
- `style.py` holds every visual decision (sizes, colours, markers) in one place.
- One `fig_*.py` per figure; each has a `build()` function.

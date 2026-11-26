# MorphoGrok

Code for the paper **"MorphoGrok: A Computable Ceiling Reveals the Limits of
Grokking in Arabic Root Extraction"** (EACL 2027).

The task is Arabic root extraction: given an undiacritized Modern Standard
Arabic lemma, spell its consonantal root. The repository contains

- the data pipeline (cleaning, root-disjoint SHA-256-frozen splits, the
  multi-form evidence conditions),
- the character-level transformer and its two training recipes (the grokking
  recipe and a conventional supervised baseline),
- the computable ceiling C_phi and its validation checks,
- evaluation (full-test re-scoring, inventory-constrained decoding, lenient
  w/y scoring, calibration), the grokking criterion, the circuit analysis,
- the off-the-shelf baselines and the figure scripts.

It contains **no data, no trained models and no result files**; see
[Data](#data).

## Install

```bash
pip install -e ".[train,figures,dev]"     # Python >= 3.10
pytest                                     # 11 unit tests, no data needed
```

The third-party baselines need extra tools, which are only installed on
request: `pip install -e ".[baselines]"`, plus the CAMeL Tools morphology
database (`camel_data -i morphology-db-msa-r13`).

All commands below are run from the repository root. Set `PYTHONHASHSEED=0`
when building splits and running the third-party baselines.

## Data

The lexicon is **Qabas** (Jarrar and Hammouda, LREC-COLING 2024), distributed by
its authors under CC BY-ND 4.0. That license does not allow redistributing
modified versions, so this repository ships neither the lexicon nor anything
derived from it (dataset files, split files, per-item predictions). Obtain
Qabas from its authors (<https://sina.birzeit.edu/qabas>) and place the CSV at
`data/raw/qabas.csv`. Everything else is rebuilt locally, and
[`experiments/split_hashes.json`](experiments/split_hashes.json) lets you check
that your splits are the ones the paper used.

```bash
python scripts/data/build_dataset.py            # Qabas CSV -> artefacts/phase0/roots_dataset.parquet
python scripts/data/make_splits.py              # L1/L2 root-disjoint splits and the train-fraction cells
python scripts/data/make_splits.py --levels L3,L4 --out-dir artefacts/phase0/splits_l34
python scripts/data/make_v2_dataset.py          # cleaned dataset + the cleaned L2 split
python scripts/data/make_root_type_map.py       # root inventory with root types
python scripts/data/make_gen2_splits.py         # Single-Form, Support-1/2, Repeat and Cross-Root
python scripts/data/make_gen2b_splits.py        # Witness and Mixed-k
python scripts/data/make_l34_clean_splits.py    # junk-free L3/L4 test sets, same root partition
python scripts/data/make_diac_splits.py         # diacritized-input cells (Appendix D)
```

Conditions of the paper and their split files
(`artefacts/phase0/splits_gen2_v2clean/L2_p30_<id>.json`):

| Paper | id | | Paper | id |
|---|---|---|---|---|
| Single-Form | `k1` | | Repeat (placebo) | `k3_self` |
| Support-1 | `k2_para` | | Cross-Root (placebo) | `k3_shuf` |
| Support-2 | `k3_para` | | Mixed-k | `kmix` |
| Witness | `k3_reveal` | | | |

## Training

The grokking recipe (full batch, AdamW, constant learning rate, no dropout, no
early stopping):

```bash
python -m grokkingroots.train.train_grok \
    --split artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json \
    --out artefacts/runs/g5_L2_k1_v2c_wd005_500k_s0 \
    --steps 500000 --weight-decay 0.05 --init-scale 1.0 --seed 0 \
    --save-ckpts --ckpt-dir artefacts/runs/g5_L2_k1_v2c_wd005_500k_s0/ckpts --resume
```

The conventional supervised baseline (recipe of Wu, Cotterell and Hulden, 2021):

```bash
python -m grokkingroots.train.train_supervised \
    --split artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json \
    --out artefacts/runs/sup_dec_do01_s0 \
    --arch dec --d-model 128 --n-layers 2 --d-ff 512 --dropout 0.1 --seed 0 --resume
```

Every run of the paper, with its split, hyperparameters and seeds, is listed in
[`experiments/`](experiments/). Both trainers resume exactly from their last
checkpoint with `--resume`. A 300-step run of the first command takes a few
minutes on a CPU and is a quick way to check an installation.

## Evaluation and analysis

| Stage | Script | What it produces |
|---|---|---|
| Accuracy | `scripts/evaluation/fulltest_eval.py` | exact match on the complete frozen test set at the validation-selected checkpoint |
| | `scripts/evaluation/dump_predictions.py` | per-item predictions (greedy; inventory-constrained with `--constrained`) |
| | `scripts/evaluation/lenient_report.py` | strict and lenient w/y accuracy with seed-level bootstrap intervals, the error taxonomy |
| | `scripts/evaluation/constrained_topk_eval.py` | inventory-constrained top-k accuracy |
| Calibration | `scripts/evaluation/calibrated_decode.py`, `conditional_calibration.py` | temperature fit on validation, risk-coverage |
| | `scripts/evaluation/ladder_posterior.py` | posterior and calibration across training |
| Ceiling | `scripts/ceiling/ceiling_method.py` | the ceiling C_phi for an evidence model (`--phi exact`, `compat` or `template`) and the accuracy its predictor realizes |
| | `scripts/ceiling/ceiling_selfcheck.py`, `ceiling_selfcheck_table.py`, `ceiling_sensitivity_table.py` | the paired validation check and the sensitivity tables |
| | `scripts/ceiling/oracle_ceiling.py`, `ambiguity_audit.py` | surface-collision ceiling; visible/hidden/junk audit of a test set |
| Baselines | `scripts/baselines/baselines.py` | the deterministic skeleton baseline |
| | `scripts/baselines/arabic_tools_baseline.py` | seven third-party systems on the same frozen test sets |
| Statistics | `scripts/analysis/aggregate_ci.py`, `g5_ci.py` | seed-level aggregation and bootstrap intervals |
| | `scripts/analysis/grok_robustness.py` | the grokking criterion and its robustness to the cutoff |
| | `scripts/analysis/count_wy_counterparts.py` | test roots with a w/y-only counterpart in the inventory |
| | `scripts/analysis/summarize_runs.py`, `plot_run.py` | per-run summaries and curves |
| Mechanism | `scripts/analysis/circuit_analysis.py` | per-bucket curves and head/FFN zero-ablation over the checkpoint ladder |
| Figures | `scripts/figures/make_all.py` | every figure of the paper, from the run directories |

The single-form ceiling of the paper (0.890, realized 0.889):

```bash
python scripts/ceiling/ceiling_method.py --phi template \
    --dataset artefacts/phase0/roots_dataset_v2clean.parquet \
    --split artefacts/phase0/splits_gen2_v2clean/L2_p30_k1.json \
    --out artefacts/camera_ready/ceiling/template.json \
    --items-out artefacts/camera_ready/ceiling/template.items.json
```

Each script documents its options in its docstring and under `--help`.

## Layout

```
src/grokkingroots/   the package: data layer, model, the two trainers, skeleton baseline
scripts/data/        dataset and split construction
scripts/evaluation/  scoring, constrained decoding, calibration
scripts/ceiling/     the computable ceiling
scripts/baselines/   skeleton and third-party systems
scripts/analysis/    statistics, grokking criterion, circuit analysis
scripts/figures/     figure scripts
experiments/         run specifications and split hashes
tests/               unit tests
artefacts/           created locally (data, splits, runs, results); never committed
```

Run identifiers keep the prefixes used during the study, which the scripts and
the experiment files rely on: `g5` is the main experiment, `g6`, `g8` and `g9`
the training-side levers, `g10` the L3/L4 strata, `g1` the earlier-protocol
runs of Appendix D, `cr` the initialization-scale runs and `sup` the supervised
baseline.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The Qabas
lexicon is not part of this repository and keeps its own license.

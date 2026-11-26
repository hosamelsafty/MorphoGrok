# Experiments

`grokking_runs.json` lists the run families trained with the grokking recipe
(`grokkingroots.train.train_grok`): the split file, the hyperparameters and the
seeds. A run directory is named `<family>_s<seed>`. The command for a family is

```bash
python -m grokkingroots.train.train_grok --split <split> --out artefacts/runs/<family>_s<seed> \
    --seed <seed> --save-ckpts --ckpt-dir artefacts/runs/<family>_s<seed>/ckpts --resume \
    <one --flag per entry of "args", with "_" written as "-">
```

| Families | Used for |
|---|---|
| `g5_L2_*` | the seven evidence conditions of the main results (5 seeds) |
| `g6_*` (auxiliary root-type loss), `g9_*` (consistency loss), `g8_L2_k1_*` (depth, width) | the training-side levers |
| `g8_L2_k3_reveal_depth*` | the circuit analysis at depth 4 and 6 |
| `g10_L3_*`, `g10_L4_*` | the geminate and quadriliteral strata |
| `g1b_L1_*`, `g1_*`, `g1b_L2_p80_*`, `g1c_*`, `g3_*`, `g4_*` | the L1 example and the earlier-protocol runs of Appendix D |

Two further groups were run outside that list.

**Initialization scale 3 on L2 Single-Form** (`cr_L2_k1_is3_v2c_wd005_500k`,
`cr_L2_k1_is3_v2c_wd01_500k`; seeds 0-4): the Single-Form command with
`--init-scale 3.0` and `--weight-decay 0.05` or `0.1`, 500,000 steps.

**Supervised baseline** (`grokkingroots.train.train_supervised`, seeds 0-4, on
`splits_gen2_v2clean/L2_p30_k1.json`):

| Run family | Flags |
|---|---|
| `sup_encdec_do01`, `sup_encdec_do03` | `--arch encdec --dropout 0.1` / `0.3` |
| `sup_dec_do01`, `sup_dec_do03` | `--arch dec --d-model 128 --n-layers 2 --d-ff 512 --dropout 0.1` / `0.3` |
| `sup_dec_do00_500k`, `sup_dec_do01_500k` | the `dec` flags with `--dropout 0.0` / `0.1` and `--max-updates 500000` |

The sweeps of Appendix D (weight decay, initialization scale, depth, train
fraction, early-stopped twins) use the same trainer on the `splits/` cells with
the values given in that appendix.

`split_hashes.json` gives, for every split file, its sizes and the SHA-256 of
its test set as stored in the file (`test_sha256`; some entries keep only the
first 12 characters). Splits rebuilt from the same lexicon must reproduce them.

# Clean multi-seed pipeline run — `EXP=20260610`

3 datasets × 2 priors × **seeds {42, 43, 44}**, **per-seed backbones** (full end-to-end
variance), **epochs 200 / patience 15**, `LR` left at default (LR-flat — see lr-find).
Everything namespaces under `EXP=20260610`: wandb group `20260610-<dataset>`,
results `results/{diffusion,eval}/20260610/...`, logs `logs/20260610/...`.

Focus metrics (supervisor): **MAE, MSE/RMSE, Winkler, coverage, CRPS, W₁, energy.**

---

## 1. Ship the code (Mac)
```
git add scripts/ src/utils/paths.py src/utils/wandb_setup.py MULTISEED_COMMANDS.md
git commit -m "Clean EXP-namespaced multi-seed pipeline: wandb groups, exp paths, full chain, CRPS aggregate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin structured-t-prior
```

## 2. Clean slate — both machines
**DAIC (daic02):**
```
cd ~/linuxhome/research-tud && git pull
scancel -u $USER
rm -rf results/diffusion/* results/eval/* results/backbone/* results/residuals/* logs/*
#  keep data/ (raw + processed splits — inputs to the backbone)
```
**Mac:**
```
rm -rf results/diffusion/* results/eval/* results/backbone/* results/residuals/* logs/*
```
**WandB:** in the dashboard, delete the old `diffusion-*` / `backbone-*` runs so only the
new `20260610-*` groups remain.

## 3. Smoke first — ONE cell, then verify before the rest
Launch a single chain and confirm the plumbing before committing all 9:
```
EXP=20260610 bash scripts/daic/submit_pipeline.sh metrla 42
```
Once the backbone starts, check:
- `squeue --me` shows `backbone … R`, then `residuals (Dependency)`, etc.
- log lands at `logs/20260610/backbone/backbone_metrla_seed42/<jobid>.out`
- wandb shows group **`20260610-metrla-backbone`** (and later `20260610-metrla` for diffusion)

If those look right, launch the rest.

## 4. Launch the full grid — one command per (dataset, seed)
Each runs the full chain: **backbone → residuals → {kron,iso} diffusion → {kron,iso} eval**.
Run them in any order (Slurm queues them):
```
EXP=20260610 bash scripts/daic/submit_pipeline.sh metrla  43
EXP=20260610 bash scripts/daic/submit_pipeline.sh metrla  44
EXP=20260610 bash scripts/daic/submit_pipeline.sh pemsbay 42
EXP=20260610 bash scripts/daic/submit_pipeline.sh pemsbay 43
EXP=20260610 bash scripts/daic/submit_pipeline.sh pemsbay 44
EXP=20260610 bash scripts/daic/submit_pipeline.sh pems08f 42
EXP=20260610 bash scripts/daic/submit_pipeline.sh pems08f 43
EXP=20260610 bash scripts/daic/submit_pipeline.sh pems08f 44
```
(metrla 42 was the smoke in step 3 — don't resubmit it.)

Each command prints every `sbatch` it fires (with the `--gres=gpu:1 --exclude=gpu[36-53]`
resource ask and the job id), so you see exactly what's queued.

## 5. Watch
```
squeue -u $USER --format='%.10i %.30j %.2t %.11M %.19R'
```
`(Dependency)` = waiting for its parent stage; a node like `gpu30` = running; never `gpu36–53`.

## 6. Pull results + aggregate (when done)
**Mac — pull this experiment's eval + logs (single lines):**
```
rsync -avz --no-perms --progress daic-exp:/tudelft.net/student-homes/a/sakin/research-tud-data/eval/20260610/ ~/Projects/research-tud/results/eval/20260610/
rsync -avz --no-perms --progress daic-exp:/tudelft.net/student-homes-linux/a/sakin/research-tud/logs/20260610/ ~/Projects/research-tud/logs/20260610/
```
**Then aggregate (mean±std over seeds, tables + figure):**
```
python scripts/aggregate_eval.py --exp 20260610
python scripts/analyze_training_logs.py --logs logs/20260610/diffusion --seed 42   # per-seed curves
```
Outputs land in `results/analysis/` (`eval_aggregate_20260610.md`, `fig_eval_aggregate_20260610.png`).

---

### Notes
- **Re-run later?** Bump `EXP` (e.g. `20260612`) → completely fresh wandb groups, results, logs. Nothing collides.
- Backbone + residuals are **shared per (dataset, seed)**, not under `EXP`; re-running the same seed reuses them unless you clear them.
- Eval runs on the **full test set** (no window cap) with CRPS — directly comparable to DiffSTG/RIPCN.

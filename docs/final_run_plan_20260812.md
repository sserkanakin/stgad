# Final-thesis run plan — 2026-08-12

Four campaigns, all approved. Every one is submitted from DAIC with the shard
mechanism so the two accounts (`sakin`, `dkrylov`) split the 8-job queue limit:
run the same command on both, one with `SHARD=a` and one with `SHARD=b`.

Before submitting anything, on each account:

```bash
cd $PROJECT_DIR && git pull            # picks up --anc-c-lo, the rsh arm, DENOISER
source scripts/daic/activate.sh
scripts/daic/check_quota.sh            # windowshome is 8G against ~10G of caches
```

---

## R2 — sampler resolution (HIGHEST PRIORITY, eval only)

**The finding it chases.** `reverse_sample_ancestral` stops at `c_lo = 0.02` and
returns `R̂₀` there, discarding the noise still present. The variance-weighted
fraction still un-denoised at that point,
`f(c_lo) = Σ var_m (1−e^{−2 θ_m c_lo}) / Σ var_m`, is **3.15% for the isotropic
control in every cell** (θ ≡ 0.8) and **19.0–28.2% for the structured prior**,
also in every cell. Training draws go down to `8e-4`, where the same figures are
0.13% and 0.9–2.4%. So the locked protocol resolves the two arms to fidelities
that differ by 6–9×, grid-wide.

**Local probe, pems08f, 256 windows, k=24, seed 42:**

| cell | arm | c_lo | coverage | CRPS | Winkler | energy |
|---|---|---|---:|---:|---:|---:|
| gcn (f = 26.1%) | iso | 0.02 | 0.751 | 0.1085 | 221.4 | 11.18 |
| | iso | 8e-4 | 0.742 | 0.1087 | 223.7 | 11.24 |
| | emp | 0.02 | 0.713 | 0.1047 | 222.1 | 11.12 |
| | emp | **8e-4** | **0.761** | **0.1041** | **210.1** | **10.62** |
| gwn (f = 19.2%) | iso | 0.02 | 0.772 | 0.0514 | 105.2 | 3.13 |
| | iso | 8e-4 | 0.770 | 0.0515 | 105.9 | 3.30 |
| | emp | 0.02 | 0.712 | 0.0518 | 112.4 | 3.82 |
| | emp | **8e-4** | **0.761** | **0.0516** | **106.7** | **3.53** |

The isotropic arm moves by under a point of coverage on both cells — its spectrum
was already resolved. The structured arm gains **4.8–4.9 points**, overtakes the
control on coverage, Winkler and energy, and keeps its CRPS advantage. Three of
the thesis's four negative findings are, on these cells, artefacts of where the
sampler stops.

Note the effect is **not** confined to stiff cells: Graph WaveNet gains as much as
the GCN, because what matters is the variance-weighted fraction above (19% vs
26%) and not the single stiffest mode.

**Submit** (pilot: 4 cells × 2 arms, ~1 h/job):

```bash
DRY=1 scripts/daic/anc_sweep.sh                 # preview first
SHARD=a scripts/daic/anc_sweep.sh               # on sakin
SHARD=b scripts/daic/anc_sweep.sh               # on dkrylov
```

Grid is `c_lo ∈ {default, 8e-4} × steps ∈ {50, 200}`, k=100, test split. The
`default × 50` combination reproduces the locked protocol byte-for-byte and is
skipped when its json already exists, so headline numbers cannot be clobbered.

**Then, if the pilot holds** (expect a coverage gain of roughly 4–6 points on the
structured arm in every cell, and under one point on the control):

```bash
CELLS_SPEC=full SHARD=a scripts/daic/anc_sweep.sh    # all 18 cells
CELLS_SPEC=full SHARD=b scripts/daic/anc_sweep.sh
```

**Read it with** `python scripts/analysis/anc_sweep_report.py`.

**Decision this run makes.** If the effect scales with `1/θ_max` as predicted,
the locked protocol is not a fair comparison between arms and the whole
calibration section of Chapter 5 has to be re-framed: the structured prior does
not under-disperse, the sampler under-resolves it. If it does not scale, the
prior explanation stands and is now defended against the obvious objection.
Either way this must land before Chapters 5–7 are finalised.

---

## R4 — seed expansion on the history grid

Three more diffusion seeds across all 18 cells × 2 priors = 108 runs. Takes the
main CRPS claim from `16/24, p = 0.08` to conclusive.

```bash
# phase A — the caches already on the machine
SEEDS="2 3 4" DATASETS="pems08f pems08s pemsbay" SHARD=a scripts/daic/histcond.sh
SEEDS="2 3 4" DATASETS="pems08f pems08s pemsbay" SHARD=b scripts/daic/histcond.sh
# phase B — after the storage rotation in docs/conditioning_report.md
SEEDS="2 3 4" DATASETS="metrla pems04f pems04s" SHARD=a scripts/daic/histcond.sh
SEEDS="2 3 4" DATASETS="metrla pems04f pems04s" SHARD=b scripts/daic/histcond.sh
```

Do **not** start phase B before phase A reports 40/40 test evals — deleting a
cache a failed cell needs strands its re-run. The script refuses to submit cells
whose cache is absent, so a partial upload cannot half-run.

---

## R5 — test evals for the two validation-only axes

Both are eval-only on checkpoints that already exist; only the eval was never
run. `--prior` now accepts `vp`, which is why the DDPM arm was never evaluable.

```bash
# corruption family: DDPM (vp) under forecast-only, comparable to iso/emp fo
for ds in pems08f pems08s pems04f pems04s; do for bb in mlp gcn gwn; do
  sbatch --export=ALL,CELLS="20260721_ladder/ddpm|$ds|$bb|vp|42" scripts/daic/eval_saved.sbatch
done; done

# denoiser capacity: 996k UGnet, prev-residual conditioning (both rungs share it)
for ds in pems08f pems08s pems04f pems04s; do for bb in mlp gwn; do
  sbatch --export=ALL,CELLS="20260721_ladder/iso-past-ugnet|$ds|$bb|isotropic|42;20260721_ladder/emp-past-ugnet|$ds|$bb|kronecker|42" scripts/daic/eval_saved.sbatch
done; done
```

**Caveat to fix, not to live with.** The UGnet capacity axis exists only under
the retired prev-residual conditioning, so the three-channel comparison in
Section 5.5 currently mixes regimes. The clean fix is one more training batch —
the capacity channel is an isotropic-arm statement and needs no prior comparison:

```bash
EXP=20260812_histcond_ugnet DENOISER=ugnet ARMS=iso SEEDS=42 \
  DATASETS="pems08f pems08s pemsbay" SHARD=a scripts/daic/histcond.sh
```

18 runs, and then the whole of Section 5.5 sits on one conditioning regime.

---

## R6 — reshaped arm under history conditioning

The tightest-fit extreme point of the structure axis currently exists only on the
forecast-only and retired arms, so Section 5.7's last ablation is stranded.

```bash
ARMS=rsh SEEDS="1 42" DATASETS="pems08f pems08s pemsbay" SHARD=a scripts/daic/histcond.sh
ARMS=rsh SEEDS="1 42" DATASETS="pems08f pems08s pemsbay" SHARD=b scripts/daic/histcond.sh
```

`histcond_cell.sbatch` now maps `rsh` to `--temporal reshaped`, matched to the
ladder's `rsh-past` cells in every other flag.

R6 also interacts with R2 and is the sharpest test of it: the reshaped prior is
the stiffest of the three arms, so if the terminal floor is what drives
under-dispersion, `rsh` should gain the most from lowering it — its coverage on
the one existing EM sweep goes 0.646 → 0.878 when the integrator is pushed to
convergence, the largest move of any arm.

---

## Ordering

1. **R2 pilot** — nothing in Chapters 5–7 is final until it reports.
2. **R5** — cheap, removes three caveats from the text.
3. **R4** — the long pole; start phase A as soon as the queue has room.
4. **R6** — after R4 phase A, same caches.
5. **R2 full grid** — once the pilot direction is known.

## Logging and provenance

Every campaign writes under `$OUTPUT_ROOT/<exp>/<ds>/<bb>/<prior>/seed<s>/`,
with `train/config.json` recording every resolved flag and
`eval/test_metrics_*.json` naming the checkpoint, sampler, step count and
(new) terminal floor. Non-default sampler settings get their own filename, so a
re-sampled run can never be mistaken for a locked-protocol one. Slurm logs go to
`$OUTPUT_ROOT/logs/<exp>/`. After each campaign:

```bash
scripts/daic/sync_from_daic.sh                       # mirror to the Mac
python scripts/analysis/config_parity.py             # flags match the reference
python scripts/analysis/thesis_tables.py             # regenerate every table
```

`thesis_tables.py` is now the single source for every number in Chapters 5–7.

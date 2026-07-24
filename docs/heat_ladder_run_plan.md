# Heat-ladder completion — what to run (2026-07-24)

Goal: make the whole experiments chapter one **heat-vs-isotropic** story (the actual
method), so the main comparison table stops using the empirical/covariance arm that the
methodology relegates to an appendix. This needs the **learned-heat forecast-only ladder**.

## State on disk (verified)

| arm | conditioning | backbones present | trained? | evaluated? |
|---|---|---|---|---|
| `iso` | fo | mlp, gcn, gwn × 4 ds | yes | **yes** (5 seeds) |
| `cartL-fo` (learned heat, γ=0.05, kNN-5) | fo | **gcn, gwn** × 4 ds (+metrla/pemsbay gwn) | **yes (2 seeds)** | **NO — never ran** |
| `cart-fo` (road heat, γ=0.8) | fo | pems08f mlp/gcn/gwn; others gcn/gwn | yes | partial |
| `cartL-fo` **mlp** | fo | — | **NO** | — |

So: the learned-heat gcn+gwn checkpoints exist and only need the eval pass; the learned-heat
**mlp** rung (the weak-backbone cell that shows the ladder law) was never trained.

`cartL-fo` arm definition (from `results/20260724_heat/cartL-fo/*/config.json`):
`prior=kronecker, spatial-graph=correlation, knn-k=5, gamma=0.05, gamma-t=0.6,
temporal=cartesian, c-0=50, schedule=fcps, loss=snr, noise-sampler=loguniform,
norm=pernode, cond_past=False, gcnn denoiser (24×3, hops 5) = 17,305 params, residual_seed=42`.

## Step 1 — eval the already-trained learned-heat cells (fast; no retraining)

Minutes/cell on A40. Uses the existing checkpoints; reads the learned graph from each
saved config. Skips mlp automatically (no checkpoint).

```bash
EXP=20260724_heat \
DATASETS="pems08f pems08s pems04f pems04s" \
BACKBONES="gcn gwn" \
ARMS="cartL-fo:kronecker" \
SEEDS="42 1" \
scripts/daic/submit_ladder_eval_array.sh          # add --dry first to preview
```

For the recalibrated column too, add the validation-fit spread grid (matches the banked
protocol's `_srecal` files): append `AUTOSPREAD="1.0,1.1,1.2,1.3,1.4,1.5"` to the env.

Writes `20260724_heat/cartL-fo/<ds>/<bb>/kronecker/seed*/eval/test_metrics_bestcrps_ancestral50.json`.

## Step 2 — train + eval the missing learned-heat MLP rung (the ladder-law cell)

Per dataset × seed (8 jobs). `EXP` carries the arm folder so it lands in the same tree.

```bash
for DS in pems08f pems08s pems04f pems04s; do for S in 42 1; do
  sbatch --partition="$DAIC_PARTITION" --gres="$DAIC_GRES_GPU" \
    --export=ALL,EXP=20260724_heat/cartL-fo,DATASET=$DS,BACKBONE=mlp,SEED=$S,PRIOR=kronecker,EVAL_STEPS=50,\
EXTRA_FLAGS="--spatial-graph correlation --knn-k 5 --gamma 0.05 --gamma-t 0.6 --temporal cartesian --c-0 50 --schedule fcps --loss snr --noise-sampler loguniform --norm pernode --repair-graph --eval-k 100" \
    scripts/daic/submit_train_eval_diffusion.sbatch
done; done
```

(If any run diverges from the banked cells, copy the exact field from a sibling
`cartL-fo/.../config.json` — that file is ground truth for the arm.)

## Step 3 — optional, to fully match the 5-seed controls

- Road-heat `cart-fo` ladder gaps for a road-vs-learned column: `pems08s/mlp, pems04f/mlp,
  pems04s/mlp, pems04s/gcn` (train+eval, same as Step 2 but `--spatial-graph road --gamma 0.8 --c-0 7`).
- Seed top-up 42/1 → +126/2022/3407 on the six winning heat cells (`docs/heat_campaign_report.md` §1)
  to harden the wins to the controls' seed count.
- `cartLw-past` (metrla winsor) / `cartLp-past` (pems04s/gcn patience) — the last heat eval
  job (dkrylov 275328) that hit walltime; pull when it drains.

## Cartesian-vs-old provenance of the ENTIRE experiments chapter (2026-07-24)

The only `temporal=cartesian` (heat) runs are in `20260724_heat`. Everything else in
`20260721_ladder`/`20260721_gammasweep` is `temporal=empirical` or `reshaped` — the OLD
method. Confirmed by reading `config.json`. So:

| chapter item | current source | Cartesian? | how to make it Cartesian |
|---|---|---|---|
| §Residual diagnostics numbers | backbone residuals | prior-agnostic | fine as-is |
| Fig heat-smoothness | heat-prior projections | ✅ yes | fine |
| Fig forward-validation | heat forward process | ✅ yes | fine (regen GAD-style per your ask) |
| §Ladder table (`tab:exp_main`) | emp/reshaped/iso **fo** | ❌ OLD | eval trained `cart-fo`/`cartL-fo` (gcn+gwn); **train+eval mlp** |
| Fig backbone-improvement (f8) | empirical fo | ❌ OLD | regen from `cart-fo` after eval |
| Fig fanchart (f2) | empirical/linear fo | ❌ OLD | regen from a `cart` run |
| §γ-sweep (`tab:exp_gamma`) | empirical γ | ❌ OLD | **eval** trained `cartg005/cartg02/cartg04-fo`+`cart-fo` (γ=0.05/0.2/0.4/0.8) — trained, 0 evaluated |
| §Conditioning (`tab:exp_subst` L) | iso/emp fo vs past | ❌ OLD | `cart-fo` vs `cart-past` already exist — re-base after eval |
| §Capacity / UGnet (`tab:exp_subst` R) | **emp-past-ugnet** (d_h32) | ❌ OLD | **no Cartesian UGnet exists — must train+eval** `cart` with `--denoiser ugnet --d-h 32` |
| Training-cost (`tab:exp_convergence`) | iso vs empirical gwn | ❌ OLD | recompute from `cart` gwn `validation_history.json` |
| §Few-step (f4) | empirical/iso fo | ❌ OLD | heat few-step val exists (heat report §4); regen |
| §Heat deployment (`tab:exp_heat`) | `cart`/`cartL` past | ✅ yes | done (best/worst envelope) |
| §Raw-signal | cartesian raw | ✅ yes but single-seed | test evals + ≥3 seeds |
| §SOTA `tab:exp_sota` "ours" row | recalibrated **empirical** | ❌ OLD | swap in `cart-past` heat CRPS |
| Fig reliability (f3) / mechanism (f1) | empirical | ❌ OLD | regen from `cart` |

**Answering the specific questions:** the UGnet ("capacity") numbers were run 2026-07-21 as
`emp-past-ugnet` — the empirical covariance variant, NOT Cartesian; there is no Cartesian
UGnet run, so that ablation must be re-run if it stays. The past/fo ladder numbers are
likewise empirical/iso except the deployment table.

### Cluster run list to make it ALL Cartesian
1. **Eval only (trained, fast):** `cart-fo`+`cartL-fo` gcn+gwn (ladder); `cartg005/02/04-fo`
   + `cart-fo` (γ-sweep) — all 0/– evaluated. Use the Step-1 eval-array with
   `ARMS="cart-fo:kronecker cartL-fo:kronecker cartg005-fo:kronecker cartg02-fo:kronecker cartg04-fo:kronecker"`.
2. **Train+eval (missing):** `cart-fo`/`cartL-fo` **mlp** rung (ladder law); `cart` **UGnet**
   d_h=32 past (capacity ablation, add `--denoiser ugnet --d-h 32`); raw-signal ≥3 seeds test.
3. **Regenerate figures from the Cartesian evals:** f8, f2, f1, f3, f4, convergence — the
   plotting scripts in `scripts/analysis/` re-point to `20260724_heat`.

## After the numbers land

1. Reconstruct `tab:exp_main` as **learned-heat (and/or road-heat) vs isotropic** across the
   backbone ladder — one prior, one protocol, one seed policy.
2. Demote the empirical/covariance variant to the appendix ablation (heat report: "can be
   dropped at zero cost").
3. Then apply the rest of `docs/audit_report_20260724.md` (appendix val→test fix, citations,
   symbols, language) against the settled numbers.

## Local fallback (directional only)

Eval runs on this Mac (MPS) but at ~2.4 h/cell full-fidelity — impractical for the full set.
A reduced read (`--max-windows 768 --k 100`) gives the ladder shape in ~20 min/cell if a
quick directional answer is wanted before the cluster run.

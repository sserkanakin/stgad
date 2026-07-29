# The conditioning axis — full picture

*2026-07-28. Auto-generated tables live in `results/analysis/conditioning/conditioning_report.md`
(regenerate with `scripts/analysis/conditioning_report.py`); this file is the narrative.*

## The question

What does the residual diffusion gain from each thing we can condition it on?

| arm | conditioning | shape | status |
|---|---|---|---|
| `fo` | `C = Ŷ` | (B,N,12) | **complete** — 20260721_ladder |
| `hist` | `C = [X \| Ŷ]` | (B,N,24) | **running** — 20260728_histcond |
| `prev` | `C = [R_prev \| Ŷ]` | (B,N,24) | **retired** — 20260721_ladder `-past` |

`X` is the observed 12-step history window the backbone consumed; `Ŷ` is its point
forecast. Both are z-scored by the signal scaler. Since P = H = 12 on all six datasets,
the history arm is a drop-in: the GCNN denoiser takes 2 conditioning channels instead of
1, 17,449 params instead of 17,305. Nothing else changes.

## The correction that prompted the rerun

The arm the thesis has been calling "history conditioning" was not that.

`--cond-past` fell back **silently** to the previous window's *residual* field whenever a
residual cache had no `history` key:

```python
if "history" in d:  past = z(d["history"])          # never taken until 2026-07-28
else:               past[H:] = R0u[:-H]             # the previous window's residual
```

No cache ever had that key — `extract_residuals.py` only started writing it in commit
`7eb0a49`, dated today. So every `-past` run in `20260721_ladder` (150 test-evaluated
runs) trained `C = [R_prev | Ŷ]`. That is a coherent ablation — the previous window's
backbone error is observable at forecast time and carries error volatility — but it is a
different experiment from conditioning on the observed traffic state, and the thesis text
describes the latter.

Fixed in `1e253cc`: the history is now read from the processed split when the cache lacks
the key (same array, same row order, so the ~10 G of caches never needed regenerating),
and a run that cannot find it **raises** instead of falling back. `tests/test_cond_history.py`
pins this.

Per the 2026-07-28 decision the `prev` arm is retired from the thesis. Its numbers stay on
disk and in the generated tables, clearly labelled.

## Result 1 — forecast-only (complete)

Locked protocol: `best_crps.pt`, ancestral@50, k=100, y==0 masked, test split. Test CRPS,
seeds paired; backbone/residual seed pinned to 42 so only the diffusion seed varies.

| dataset | iso/mlp | kron/mlp | iso/gcn | kron/gcn | iso/gwn | kron/gwn |
|---|---|---|---|---|---|---|
| pems08f | 0.2295 | 0.1978 | 0.1057 | 0.1050 | 0.0497 | 0.0502 |
| pems08s | 0.0430 | 0.0428 | 0.0248 | 0.0246 | 0.0174 | 0.0175 |
| pems04f | 0.2790 | 0.2606 | 0.1158 | 0.1169 | 0.0663 | 0.0668 |
| pems04s | 0.0506 | 0.0508 | 0.0289 | 0.0288 | 0.0205 | 0.0206 |
| metrla  | 0.0948 | 0.0918 | — | — | 0.0432 | 0.0443 |
| pemsbay | 0.0570 | 0.0545 | — | — | 0.0209 | 0.0212 |

`kron` = Σ_T ⊗ Σ_S with Σ_S = (L_road + 0.8·I)⁻¹ (GAD's fixed γ) and Σ_T the empirical
horizon covariance (ridge ρ = 0.1).

The structured prior's advantage tracks how weak the backbone is, and only there:

| backbone | kron vs iso, CRPS |
|---|---|
| mlp (312 params) | −13.8% / −6.6% / −4.4% / −3.2% on pems08f / pems04f / pemsbay / metrla; ≈0 on the two speed-flow pairs |
| gcn (17k) | between −0.8% and +0.9% — noise |
| gwn (309k) | +0.5% to +2.5% — iso wins everywhere |

This is the ladder law the thesis already reports, now on a complete forecast-only grid.

## Result 2 — history + forecast

`20260728_histcond`: iso and kron × 6 datasets × 3 backbones × 2 seeds, same 17k
denoiser, same locked protocol. **COMPLETE** — 80/80 cells (both phases), every job exit
0:0, zero failures, config parity verified field-by-field against the ladder on every cell.

CRPS change vs forecast-only, paired on seed (iso / kron in each cell):

| dataset | mlp | gcn | gwn |
|---|---|---|---|
| pems08f | −60.1 / −55.7 | −29.5 / −29.8 | +0.1 / +0.2 |
| pems08s | −35.6 / −37.7 | −11.7 / −11.9 | +1.0 / −0.9 |
| pems04f | −56.0 / −55.4 | −18.8 / −20.7 | +0.0 / +0.1 |
| pems04s | −36.8 / −37.5 | −9.9 / −10.5 | −0.0 / −0.3 |
| metrla | −28.8 / −25.7 | −10.3 / −11.8 | −0.6 / −1.1 |
| pemsbay | −41.2 / −39.6 | −23.1 / −23.4 | −0.6 / −0.5 |

Rolled up over all six datasets:

| backbone | iso | kron | cells improved |
|---|---|---|---|
| mlp (312) | **−43.1%** | **−41.9%** | 12/12 |
| gcn (17k) | **−17.2%** | **−18.0%** | 12/12 |
| gwn (309k) | −0.0% | −0.4% | 7/12 |

**The gain is inversely proportional to backbone capacity** — unanimous on mlp and gcn,
nil on gwn — and almost independent of the prior (iso and kron agree within ~2 points in
all 18 cells). Conditioning and prior act on different margins; gwn has already extracted
the history, so repeating it to the denoiser is redundant.

**It halves the structured prior's advantage** — the substitution story with the real
history (kron-minus-iso, mean over 6 datasets, cells won in brackets):

| backbone | forecast-only | history+forecast |
|---|---|---|
| mlp | **−4.7%** (5/6) | −1.5% (5/6) |
| gcn | +0.2% (3/6) | −0.7% (5/6) |
| gwn | +1.1% (0/6) | +0.8% (1/6) |

Against the measured **1.0 pp seed-noise floor** (median seed-to-seed range of the
kron-vs-iso delta), only the mlp/forecast-only column clears noise: −13.8 pems08f,
−6.6 pems04f, −4.4 pemsbay, −3.2 metrla, every seed negative. Supplying the history cuts
that mean advantage from −4.7% to −1.5%. The gcn sign flip (+0.2 → −0.7, 5/6 cells) is
suggestive but inside noise at 2 seeds. On gwn iso wins 0/6 and 1/6 with per-seed ranges
as tight as 0.2 pp — small, but reproducible rather than noise.

**History is NOT a relabelling of the retired prev-residual arm.** The two differ
substantially and not in one direction: history is better on every gcn cell
(pems08f −29.5% vs −19.1%), worse on pems08s/mlp (−35.6% vs −43.2%), and slightly worse on
pems08f/mlp (−60.1% vs −63.9%). The old numbers cannot be carried over.

- **Phase A** (pems08f, pems08s, pemsbay): 40/40 cells. **Done.**
- **Phase B** (metrla, pems04f, pems04s): 40/40 cells. **Done.** Needed a storage
  rotation — windowshome has an 8 G quota against ~10 G of caches.

  Rotation, in order (do **not** start before the prior phase reports 40/40 test evals — a
  failed cell needs its cache to re-run, and deleting it first strands the re-run):

  1. confirm `find …/20260728_histcond -name 'test_metrics_bestcrps_ancestral50.json' | wc -l`
     is 40 and no log contains `TRAIN FAILED` / `EVAL FAILED`;
  2. `rm -rf` the `pems08f`, `pems08s`, `pemsbay` dirs under
     `~/windowshome/research-tud-data/residuals/` (frees ≈5.4 G; the Mac keeps the
     full 9.7 G copy, so this is recoverable);
  3. rsync `metrla`, `pems04f`, `pems04s` residual caches up (≈4.5 G) — **with
     `--inplace`**: windowshome rejects rsync's temp files with `mkstemp … Operation not
     permitted`, and without it the transfer reports success while copying nothing;
  4. `DATASETS="metrla pems04f pems04s" scripts/daic/histcond.sh` — the script refuses
     to submit any cell whose cache is absent, so a partial upload cannot half-run;
  5. `CONDS=fo DATASETS=metrla BACKBONES=gcn scripts/daic/histcond.sh` for the last
     forecast-only hole;
  6. re-run `config_parity.py` against `results/20260721_ladder` once the first configs
     land, **before** the queue drains.

  Sequential on purpose. Uploading pems04s early to overlap with phase A would leave
  ~1.1 G of headroom for both campaigns' outputs, and a quota exhaustion mid-run fails
  jobs silently — not worth the couple of hours saved.

Every hyperparameter is pinned to `20260721_ladder`, verified by a per-cell preflight that
asserts the conditioning is 2H wide, that its forecast half is bit-identical to the
forecast-only arm, that the target is unchanged, and that the history half really is the
observed window. The flags that differ from the script defaults are passed explicitly —
notably `--schedule fcps`, because the default `auto` would put the isotropic arm on ULS
and the kronecker arm on FCPS and confound the prior comparison with a schedule change.

## Result 3 — prev-residual (retired, for the record)

Conditioning on the previous window's residual was worth far more than any prior:
−48% (iso) / −46% (kron) mean CRPS on mlp, −10% on gcn, ≈−0.7% on gwn; and it erased the
structured prior's mlp advantage entirely (pems08f mlp: kron −13.8% vs iso under `fo`,
+1.4% under `prev`). That is the substitution story — capacity, conditioning and prior are
competing channels for the same information, and the learned channels dominate. Whether it
survives with the *actual* history is what phase A/B answers.

## Verification figures

`scripts/analysis/forward_process_figures.py`, on the prior above:

- `results/analysis/forward_process/stationary_<ds>_gwn.png` — the forward process lands on
  the stationary law it was designed for: sampler vs closed form σ²/θ over all N·H modes
  (rel. error 0.049–0.053 across datasets), convergence vs c̄ with c̄_hi = 5/θ_min marked,
  the cascade, and how prior draws compare to the residuals in **each factor separately**.
  Σ_T is fit, so per-horizon variance tracks the data in shape (level off by 0.83–1.14×
  from the ρ=0.1 ridge and trace matching); Σ_S is imposed, so it is a smooth heat-kernel
  envelope over a ragged empirical spectrum. In the node domain the mismatch **changes
  sign by dataset** — on pems08f the prior asserts *more* 1-hop coupling than the residual
  has (0.19 vs 0.14), on metrla *less* (0.07 vs 0.15) — so the panel reads the direction
  off the numbers instead of asserting one. All of this is labelled as what it is rather
  than presented as a match.
- `results/analysis/forward_process/gad_forward_<ds>_gwn.png` — GAD Fig. 1 on our θ: mean
  decay `e^{−θc̄_t}` and covariance growth `(σ²/θ)(1−e^{−2θc̄_t})` per eigenvalue under FCPS
  vs ULS at **matched total heat** (ULS slope 2c₀ ⇒ c̄(T) = c₀ = 7), coloured across the
  whole spectrum so the eigenvalue cascade is visible rather than sampled at five points.
  FCPS holds the slow graph modes to t ≈ 0.6 while ULS has erased them by t ≈ 0.3.

## Calibration, and why it may be a sampler artifact

Under the locked ancestral@50 protocol the structured prior under-covers in all 18 cells
(0.70–0.79 vs iso's 0.77–0.83, nominal 0.90). The one matched EM step sweep on disk
(pems08f/gwn/seed42, iso vs emp vs rsh at identical steps) says much of that is the
sampler, not the prior:

| sampler | iso CRPS | emp CRPS | emp−iso | iso cov | emp cov | rsh cov |
|---|---|---|---|---|---|---|
| ancestral@50 | 0.04966 | 0.05010 | +0.9% | 0.822 | 0.764 | 0.646 |
| EM@1000 | 0.04953 | 0.04957 | +0.1% | 0.857 | 0.860 | — |
| EM@1500 | 0.04953 | 0.04957 | +0.1% | 0.856 | 0.855 | 0.878 |

EM's stability floor scales as n ≳ θ_max·c₀. The structured prior is stiff (θ_max 16 here,
94–152 on the gcn cells) while iso is flat (θ_max = 0.8), so at fixed steps EM integrates
the structured arm far less accurately. Push it to convergence and the accuracy gap closes
to a **tie** (+0.1%, inside noise) while iso barely moves — and the calibration gap closes
entirely (emp 0.764 → 0.855, matching iso; rsh 0.646 → 0.878, the *best* of the three).

**Caveat: one cell, one seed, on the strongest backbone** — where both priors nearly tie
anyway. Every other EM eval on disk is iso-only, so it cannot be paired. The untested and
more interesting regime is mlp/gcn, which are stiffer and where structure actually wins.
`scripts/daic/em_sweep.sbatch` runs that (eval-only on saved checkpoints, ~3.3 h/cell at
1000 steps vs 10 min for ancestral@50 — 29×). Written, not queued.

## Run health

All 349 test-evaluated runs in `20260721_ladder` terminated cleanly (212 early-stopped, 137
ran the full 200 epochs); none were walltime-killed. The two grid holes — metrla/gcn and
pemsbay/gcn — were cancelled mid-training on 2026-07-23 and are being refilled.
Config parity across the conditioning arms was checked on all 178 matched pairs: identical
on every key except `residual_seed`, which is `None` in 9 legacy configs and resolves to
the same 42.

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

## Result 2 — history + forecast (running)

Submitted 2026-07-28 as `20260728_histcond`: iso and kron × 6 datasets × 3 backbones ×
2 seeds, same 17k denoiser, same locked protocol.

- **Phase A** (pems08f, pems08s, pemsbay — caches already on DAIC): 18 jobs, 36 runs,
  plus 2 jobs filling the pemsbay/gcn forecast-only hole.
- **Phase B** (metrla, pems04f, pems04s): needs a storage rotation first — windowshome has
  an 8 G quota against ~10 G of caches, so phase A's caches come off before these go on.

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
  Σ_T is fit, so per-horizon variance tracks the data in shape; Σ_S is imposed, so it is a
  smooth heat-kernel envelope over a ragged empirical spectrum, and in the node domain the
  prior asserts more 1-hop coupling (0.19) than the residual has (0.14). Both are labelled
  as what they are rather than presented as matches.
- `results/analysis/forward_process/gad_forward_<ds>_gwn.png` — GAD Fig. 1 on our θ: mean
  decay `e^{−θc̄_t}` and covariance growth `(σ²/θ)(1−e^{−2θc̄_t})` per eigenvalue under FCPS
  vs ULS at **matched total heat** (ULS slope 2c₀ ⇒ c̄(T) = c₀ = 7), coloured across the
  whole spectrum so the eigenvalue cascade is visible rather than sampled at five points.
  FCPS holds the slow graph modes to t ≈ 0.6 while ULS has erased them by t ≈ 0.3.

## Run health

All 349 test-evaluated runs in `20260721_ladder` terminated cleanly (212 early-stopped, 137
ran the full 200 epochs); none were walltime-killed. The two grid holes — metrla/gcn and
pemsbay/gcn — were cancelled mid-training on 2026-07-23 and are being refilled.
Config parity across the conditioning arms was checked on all 178 matched pairs: identical
on every key except `residual_seed`, which is `None` in 9 legacy configs and resolves to
the same 42.

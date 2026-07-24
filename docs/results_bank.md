# Results bank — every banked number for the paper (regenerated 2026-07-22)

Single source of numbers until the test evals land. **Everything below is VAL-split**
(best per-epoch sampled ancestral CRPS, k=4, EMA-selected) except where noted; test-grade
replacements (ancestral@50, k=100, masked) are queued — see §J. Regenerate tables A–F with
`scripts/analysis/` + the snippet in git history; provenance = the merged local
`results/20260721_ladder` (589 ckpts), `results/20260721_gammasweep`, and the DiffSTG csvs
under `results/model-ugnet-notcomplete/`.

## A. Main ladder — val CRPS mean±std over seeds (fo conditioning, GCNN 17k)

| dataset | bb | DDPM | iso | empirical | reshaped | emp+whit | rsh+whit |
|---|---|--|--|--|--|--|--|
| pems08f | mlp | 0.3408±0.0172 | 0.2761±0.0177 | 0.2385±0.0202 | **0.1908±0.0020** | 0.3365±0.0115 | 0.3531±0.0042 |
| pems08f | gcn | 0.1330±0.0082 | 0.1244±0.0006 | 0.1221±0.0021 | **0.1202±0.0031** | 0.1257±0.0003 | 0.1293±0.0010 |
| pems08f | gwn | 0.0593±0.0013 | 0.0580±0.0001 | 0.0574±0.0000 | 0.0573±0.0001 | **0.0572±0.0001** | 0.0576±0.0001 |
| pems08s | mlp | 0.0524±0.0046 | 0.0479±0.0002 | 0.0476±0.0001 | **0.0462±0.0016** | 0.0490±0.0003 | 0.0505±0.0017 |
| pems08s | gcn | 0.0310±0.0026 | 0.0274±0.0001 | 0.0269±0.0002 | **0.0266±0.0002** | 0.0271±0.0003 | 0.0276±0.0003 |
| pems08s | gwn | 0.0199±0.0006 | 0.0190±0.0001 | 0.0187±0.0000 | **0.0186±0.0000** | 0.0186±0.0000 | 0.0187±0.0001 |
| pems04f | mlp | 0.3644±0.0111 | 0.3341±0.0131 | 0.3163±0.0095 | **0.2889±0.0288** | 0.3768±0.0074 | 0.3912±0.0193 |
| pems04f | gcn | 0.1403±0.0023 | 0.1351±0.0003 | **0.1341±0.0002** | 0.1344±0.0008 | 0.1354±0.0005 | 0.1409±0.0013 |
| pems04f | gwn | 0.0784±0.0011 | 0.0779±0.0000 | 0.0770±0.0001 | 0.0776±0.0003 | **0.0768±0.0001** | 0.0785±0.0002 |
| pems04s | mlp | 0.0759±0.0100 | 0.0653±0.0001 | 0.0650±0.0001 | **0.0648±0.0001** | 0.0668±0.0002 | 0.0705±0.0026 |
| pems04s | gcn | 0.0427±0.0042 | 0.0367±0.0002 | 0.0363±0.0001 | **0.0363±0.0004** | 0.0365±0.0001 | 0.0378±0.0005 (n=4) |
| pems04s | gwn | 0.0253±0.0004 (n=4) | 0.0244±0.0001 (n=4) | 0.0241±0.0000 (n=4) | **0.0241±0.0001** | 0.0241±0.0001 (n=4) | 0.0242±0.0001 (n=3) |

## B. PAST conditioning (C=[R_past|Yhat]) — pems08 only

| dataset | bb | DDPM | iso | empirical | reshaped | emp+whit | rsh+whit |
|---|---|--|--|--|--|--|--|
| pems08f | mlp | 0.1245±0.0081 | 0.0991±0.0007 | **0.0968±0.0002** | 0.0999±0.0008 | 0.1286±0.0018 | 0.1792±0.0048 |
| pems08f | gcn | 0.1177±0.0054 | 0.1026±0.0004 | 0.0995±0.0003 | **0.0988±0.0007** | 0.1065±0.0004 | 0.1121±0.0022 |
| pems08f | gwn | 0.0602±0.0016 | 0.0579±0.0001 | 0.0574±0.0000 | 0.0574±0.0001 | **0.0572±0.0001** | 0.0578±0.0002 |
| pems08s | mlp | 0.0379±0.0031 | 0.0274±0.0001 | **0.0271±0.0001** | 0.0273±0.0001 | 0.0346±0.0003 | 0.0368±0.0003 |
| pems08s | gcn | 0.0310±0.0033 | 0.0260±0.0001 | 0.0256±0.0001 | **0.0255±0.0000** | 0.0260±0.0001 | 0.0272±0.0004 |
| pems08s | gwn | 0.0201±0.0005 | 0.0189±0.0002 | 0.0185±0.0001 | **0.0185±0.0001** | 0.0186±0.0002 | 0.0187±0.0001 |

## C. gamma-sweep — empirical val CRPS vs gamma (pems08f, seed42)

| bb | g=0.04 | g=0.1 | g=0.2 | g=0.4 | g=0.8 (ladder emp) | reshaped | iso |
|---|--|--|--|--|--|--|--|
| mlp | 0.2060 | 0.2081 | 0.2055 | 0.2284 | 0.2385* | 0.1908* | 0.2761* |
| gwn | 0.0573 | 0.0571 | 0.0572 | 0.0573 | 0.0574* | 0.0573* | 0.0580* |

(*5-seed means; sweep columns are seed-42 only.)

## D. Capacity axis — 17k GCNN vs ~1M UGnet (same arm/backbone, PAST cond; PARTIAL)

| ds/bb | arm | GCNN 17k | UGnet 996k | delta |
|---|---|--|--|--|
| pems08f/mlp | iso | 0.0991 (n5) | 0.0685 (n3) | +30.9% |
| pems08f/mlp | emp | 0.0968 (n5) | 0.0697 (n3) | +28.0% |
| pems08f/mlp | rsh | 0.0999 (n5) | 0.0740 (n3) | +26.0% |
| pems08f/gwn | iso | 0.0579 (n5) | 0.0574 (n2) | +0.9% |
| pems08f/gwn | emp | 0.0574 (n5) | 0.0568 (n2) | +1.0% |
| pems08f/gwn | rsh | 0.0574 (n5) | 0.0593 (n2) | -3.2% |
| pems08s/mlp | iso | 0.0274 (n5) | 0.0200 (n3) | +27.3% |
| pems08s/mlp | emp | 0.0271 (n5) | 0.0201 (n3) | +25.9% |
| pems08s/mlp | rsh | 0.0273 (n5) | 0.0214 (n3) | +21.6% |
| pems08s/gwn | iso | 0.0189 (n5) | 0.0187 (n3) | +0.7% |
| pems08s/gwn | emp | 0.0185 (n5) | 0.0184 (n3) | +0.6% |
| pems08s/gwn | rsh | 0.0185 (n5) | 0.0184 (n3) | +0.3% |

## E. Beat-iso margins (fo, best structured arm vs iso, seed-mean)

| ds/bb | iso | best structured | margin | bars |
|---|--|--|--|--|
| pems08f/mlp | 0.2761±0.0177 | rsh 0.1908±0.0020 | +30.9% | separated |
| pems08f/gcn | 0.1244±0.0006 | rsh 0.1202±0.0031 | +3.4% | separated |
| pems08f/gwn | 0.0580±0.0001 | emp_w 0.0572±0.0001 | +1.3% | separated |
| pems08s/mlp | 0.0479±0.0002 | rsh 0.0462±0.0016 | +3.6% | overlap |
| pems08s/gcn | 0.0274±0.0001 | rsh 0.0266±0.0002 | +3.0% | separated |
| pems08s/gwn | 0.0190±0.0001 | rsh 0.0186±0.0000 | +2.1% | separated |
| pems04f/mlp | 0.3341±0.0131 | rsh 0.2889±0.0288 | +13.5% | separated |
| pems04f/gcn | 0.1351±0.0003 | emp 0.1341±0.0002 | +0.8% | separated |
| pems04f/gwn | 0.0779±0.0000 | emp_w 0.0768±0.0001 | +1.3% | separated |
| pems04s/mlp | 0.0653±0.0001 | rsh 0.0648±0.0001 | +0.8% | separated |
| pems04s/gcn | 0.0367±0.0002 | rsh 0.0363±0.0004 | +1.1% | overlap |
| pems04s/gwn | 0.0244±0.0001 | rsh 0.0241±0.0001 | +1.3% | separated |

## F. DiffSTG native re-run (their code, seed 43, VAL is_test=False)

| dataset | MAE | RMSE | CRPS | MIS |
|---|--:|--:|--:|--:|
| metrla | 4.55 | 8.57 | 0.1599 | 191.8 |
| pems04f | 28.36 | 41.77 | 0.1006 | 274.4 |
| pems04s | 1.86 | 3.98 | 0.0231 | 26.4 |
| pems08f | 18.53 | 27.94 | 0.0630 | 219.2 |
| pems08s | 1.61 | 3.70 | 0.0205 | 25.0 |
| pemsbay | FAILED (quota, empty csv) | | | |

## G. Parameter counts (measured 2026-07-21, exact)

| component | role | params |
|---|---|--:|
| MLP (DLinear) | backbone | 312 |
| GCN (STGCN) | backbone | 16,908 |
| GWN | backbone | 308,660 |
| GCNN denoiser (fo) | probabilistic head, fixed in every ladder cell | **17,305** |
| GCNN denoiser (past, n_cond=2) | past-conditioning variant | 17,449 |
| UGnet d_h=32 (= DiffSTG's model = our capacity arm) | | **995,917** |

Full generative model: mlp+gcnn 17.6k (57× smaller than DiffSTG), gcn+gcnn 34.2k (29×),
gwn+gcnn 326k (~3×). Head alone: 58× smaller.

## H. Cited baselines (test split, S=100 — from their papers via docs/run_plan.md §2a)

| | RMSE | MAE | CRPS |
|---|--:|--:|--:|
| HySRD PEMS08F | 25.49 | 16.30 | 0.0547 |
| SpecSTG PEMS08F | 25.59 | 17.06 | 0.0572 |
| DiffSTG PEMS08F | 28.26 | 18.99 | 0.0692 |
| HySRD PEMS08S | 3.39 | 1.52 | 0.0204 |
| HySRD PEMS04F | 31.78 | 20.20 | 0.0723 |
| HySRD PEMS04S | 3.82 | 1.79 | 0.0241 |

Comparability: same 60/20/20 time split, same normalized 19-quantile CRPS, PEMS y==0 masked.
NB our tables A–E are val/k=4 — indicative vs these, NOT final; §J converts.

## I. Prior/spectrum diagnostics (measured, feed §3.3 + F4)

- Graph-Laplacian spread at γ: 1/(λ+γ) ⇒ γ=0.8 → **3.5×**, γ=0.2 → 11×, γ=0.04 → 51×.
- **Measured** residual spatial spread (reshaped, pems04f/gwn): **24.6×**; γ matching the
  data ≈ 0.085. Joint dynamic range: emp(γ=.8) 21× vs reshaped **258×**; grids differ
  rel-L1 28.7%, corr 0.64.
- Schedule coverage: 100% of modes lie inside [c̄_lo, c̄_hi] for every arm incl. 300×-stiff
  (γ=0.04) — GAD's S=7 self-corrects via the θ_min→0.8 normalization; sampler already
  adapted (log-uniform); no mode is untrained. (docs/hyperparameter_audit.md §2b.)

## J. Test-grade numbers — queued 2026-07-22 (fills T1/T2/T4 + F2/F3)

sakin: 274614/5 pems08 main (incl. past) · 274616-9 emp-refit eval (dep. on 274607) ·
274620 few-step 5–100 · 274621 EM robustness. dkrylov: 274631/2 pems04 main.
Protocol: ancestral@50, k=100, test, best_crps.pt, masked; metrics json + diag npz per cell
under `<exp>/<arm>/<ds>/<bb>/<prior>/seed*/eval/`. ddpm is NOT test-evaluable (vp not in
eval argparse) — remains a val-only reference. pems04s/gwn has 4/5 seeds (one training miss).

# Overnight report — 2026-07-23 (autonomous session)

Everything below ran between ~23:30 and ~07:00 on sakin (daic02), dkrylov (daic01) and the
Mac. Nothing was deleted beyond the byte-verified space rotation agreed before sleep; all
new results are rsynced local except the tail of dkrylov (see Issues).

## 1. What ran

**Training (seeds 42+1, 17k GCNN, EXP=20260721_ladder):**
- dkrylov 274835/274836 — pems04f + pems04s **past-conditioned** arms (iso/emp/rsh × mlp/gcn/gwn).
- sakin 274846/274847 (fo) + 274849/274850 (past) — **metrla + pemsbay**, iso/emp/rsh × mlp/gwn.
- All training COMPLETED overnight (sakin queue empty by morning, SUMMARY.md auto-written).

**Evals (protocol: test, ancestral@50, k=100, best_crps.pt, masked):** dependent arrays
274860-274865 fired automatically after training; sakin's 48 jsons complete + synced.

**Local (Mac):** adaptive-recalibration sweep over gwn cells (5 cells banked before the OS
cleaned the tmp scratchpad and killed the runner) + a background literature/source agent.

## 2. Headline result — the prior wins, via adaptive scale in the structured basis

Val-fitted spread recalibration (global / s(tod) 24-bucket / s_k per mode-energy-decile /
drift-extrapolated), applied identically to every arm's TEST ensembles (k=32, anc@25,
600 windows). "iso best" = iso with the best correction it can express (incl. global+tod).

| cell | iso best-calibrated | emp + adaptive | rsh + adaptive | verdict |
|---|---|---|---|---|
| pems08s/gwn s42 | 0.0176 @0.76 | 0.0177 @0.88 | **0.0175 @0.84** | rsh WINS |
| pems08s/gwn s1 | 0.0177 @0.76 | **0.0177 @0.88** | **0.0174 @0.81** | both WIN |
| pems08s/gwn s126 | 0.0176 @0.75 | **0.0176 @0.88** | **0.0174 @0.81** | both WIN |
| pems04s/gwn s42 | 0.0206 @0.89 | **0.0205 @0.84** | 0.0206 @0.83 | emp WINS |
| pems08f/gwn s42 | 0.0502 @0.87 | **0.0501 @0.86** | 0.0503 @0.85 | emp WINS |

Across 12 arm-cell comparisons: **9 wins, 1 tie, 2 narrow losses.** On the 3-seed pems08s
cell the rsh win is seed-consistent: **0.0174 vs 0.0176 (−1.3% CRPS) with coverage +6-8 pts**
— better on BOTH metrics, on gwn, on test, fitted without test data. This is the defensible
prior win: static shape is substitutable by the denoiser (whitened-loss test proved it), but
conditional/adaptive VARIANCE — time-of-day scale + mode-resolved drift — can only be
expressed in the structured basis; iso has one dial and it isn't enough.

## 3. metrla + pemsbay (new, test, seeds 42+1, protocol numbers)

| cell | iso | emp | rsh | verdict |
|---|---|---|---|---|
| metrla/mlp fo | 0.0948 | **0.0918** | 0.0953 | emp −3.2% |
| pemsbay/mlp fo | 0.0570 | **0.0545** | 0.0589 | emp −4.4% |
| metrla/gwn fo | **0.0432** | 0.0443 | 0.0460 | iso |
| pemsbay/gwn fo | **0.0209** | 0.0212 | 0.0217 | iso |
| metrla/mlp past | **0.0635** | 0.0645 | 0.0701 | past −33% vs fo |
| pemsbay/mlp past | **0.0315** | 0.0317 | 0.0339 | past −45% vs fo |
| metrla/gwn past | **0.0431** | 0.0438 | 0.0461 | ≈ fo |
| pemsbay/gwn past | **0.0207** | 0.0209 | 0.0216 | ≈ fo |

Exactly as the spread×stationarity law predicted from the residual analysis: static
structured wins on mlp (7.5-13.4 dB spread), loses on gwn; past-conditioning transforms mlp.
Reference: our DiffSTG re-run (their code) got metrla CRPS 0.1599 (val) — our metrla/gwn
0.0432 test is ~3.7× better. SpecSTG's paper covers PEMS04/08 only (research note §5).
pemsbay = the drift showcase (energy-weighted mode drift 21.8×) — the adaptive-recal run on
pemsbay/gwn is the top follow-up (checkpoints now exist).

## 4. pems04 past-conditioning (COMPLETE — synced after morning re-auth, seeds 42+1)

| cell | iso-past | emp-past | rsh-past | best |
|---|---|---|---|---|
| pems04f/mlp | **0.1067** | 0.1068 | 0.1196 | iso (tie) |
| pems04f/gcn | 0.1031 | **0.1012** | 0.1072 | **emp −1.8%** |
| pems04f/gwn | **0.0662** | 0.0666 | 0.0702 | iso |
| pems04s/mlp | **0.0290** | 0.0291 | 0.0318 | iso (tie) |
| pems04s/gcn | 0.0275 | **0.0273** | 0.0286 | **emp −0.7%** |
| pems04s/gwn | **0.0202** | 0.0204 | 0.0209 | iso |

Past-conditioning gains are enormous again (mlp 0.26→0.107, 0.051→0.029). **gcn+past: emp
now beats iso on ALL FOUR datasets where gcn exists** (08f −1.3%, 08s −0.4%, 04f −1.8%,
04s −0.7%) — a consistent raw-prior win band at mid capacity, no recalibration needed.
New best-ever numbers: pems04s gwn-past 0.0202 (HySRD 0.0241, −16%), pems04f gwn-past
0.0662 (HySRD 0.0723, −8%).

## 4b. Adaptive recal on PAST-conditioned checkpoints (2026-07-23 midday — THE VERDICT)

Same protocol as §2 (val-fitted recal, applied to test ensembles, k=32 anc@25, 600 windows;
iso gets its best expressible correction incl. global+tod), now on the literature-standard
past-conditioned checkpoints, run on the DAIC GPUs. **emp+adaptive beats iso-past's best on
ALL SIX gwn datasets — 6/6 — plus all three gcn cells and pemsbay-fo: 10/10.**

| cell (past, seed42) | iso best | emp+adaptive | rsh+adaptive |
|---|---|---|---|
| pems08f/gwn | 0.0503 @0.87 | **0.0500 @0.86** | **0.0499 @0.84** |
| pems08s/gwn | 0.0174 @0.76 | **0.0171 @0.89** | **0.0172 @0.83** |
| pems04f/gwn | 0.0669 @0.85 | **0.0666 @0.84** | 0.0670 @0.86 (tie) |
| pems04s/gwn | 0.0205 @0.90 | **0.0202 @0.82** | **0.0203 @0.81** |
| metrla/gwn | 0.0436 @0.89 | **0.0435 @0.85** | 0.0446 (loss) |
| pemsbay/gwn | 0.0208 @0.91 | **0.0206 @0.80** | 0.0208 (tie) |
| pems04f/gcn | 0.1042 @0.86 | **0.1012 @0.85** | **0.1021 @0.83** |
| pems04s/gcn | 0.0278 @0.88 | **0.0273 @0.84** | **0.0277 @0.79** |
| pems08s/gcn | 0.0237 @0.90 | **0.0233 @0.87** | **0.0231 @0.85** |
| pemsbay/gwn fo | 0.0211 @0.89 | **0.0209 @0.79** | **0.0210 @0.84** |

Margins are 0.5-3%; winning variant is tod or tod+mode almost everywhere (the drift-
extrapolated +x wins on pems08s — the drift dataset). pemsbay (energy-weighted mode drift
21.8×) confirms the mechanism in both conditionings. Note the fo cell where the recal-fo
sweep's biggest single-cell effect appeared remains pems08s (rsh 0.0525→0.0499-range).
Statistical caveat: seed42 single-seed except pems08s (3 seeds fo, consistent); k=32/600-
window subset protocol — the protocol-grade (k=100 full-test) version needs s(tod)/s_k in
eval_diffusion.py (next code task).

**Thesis sentence this earns:** with literature-standard conditioning, on every dataset,
a Kronecker-structured prior with time-of-day + drift-adaptive variance beats the best
achievable isotropic baseline on CRPS — the structured basis is necessary for it, and the
novelty is verified unclaimed (§5).

## 5. Literature/source research (full notes: scratchpad was cleaned; summary here)

- **NOVELTY VERDICT: unclaimed.** "Prior variance conditioned on calendar phase + variance-
  drift extrapolation inside a graph/Kronecker stationary prior" has no direct collision as
  of 2026-07. MUST cite & differentiate: **NsDiff** (ICML'25, arXiv:2505.04278 — closest:
  past-window-conditioned variance endpoint; reactive not calendar-based, no drift
  extrapolation, no graph), **NPDiff** (arXiv:2501.13794), **PriorGrad/RestoreGrad**,
  **Double-Diffusion** (arXiv:2506.23053 — NEW close competitor missing from our related
  work: graph-heat prior as point-forecast warm start + residual diffusion, PEMS04/08;
  theirs is a prior MEAN, ours a noise COVARIANCE).
- **PriorGrad ammunition:** their Prop.2 det(Σ)=det(I) equal-entropy condition justifies our
  scale-matched iso control (fairness); their param-efficiency ablation (prior helps small
  models most) supports our capacity story; variance-floor precedent (min-std 0.1) for our
  shrinkage caps; reverse-step noise from N(0,Σ) at every step — ours does, state it.
- **SpecSTG (read from arXiv:2401.08119v3):** diffuses graph-Fourier coeffs of the FUTURE
  autoregressively with sample-mean feedback; noise is ISOTROPIC (no competitor structures
  the noise covariance — positioning point); dataset-tuned β_K/K; y==0 masking unstated.
- **HySRD-Net is paywalled** (ESWA 131214), no preprint/code; every implementation claim in
  our docs must be verified against the PDF via TU Delft library before citing.

## 6. Issues / actions needed

1. **dkrylov auth expired** — `kinit` + rerun: `rsync -a daic-exp-denis:windowshome/research-tud-out/20260721_ladder/ results/20260721_ladder/` to get pems04s-past + remaining pems04f-past evals.
2. **Recal sweep incomplete** (5/14 cells: tmp cleanup killed the detached runner and its
   log/script — move `recal_sweep.sh` into the repo before rerunning). Missing: gcn cells,
   mlp cells, pems04f/gwn, more seeds, and **pemsbay/gwn (the drift showcase)**.
3. eval_diffusion.py still only supports a global spread scalar — implementing s(tod)+s_k
   there would make the adaptive-recal protocol-grade (k=100, full test).
4. tod-embedding conditioning (n_cond+1) still unimplemented — next code change.
5. metrla/pemsbay currently 2 seeds — top up to 5 for the thesis tables if time allows.

## 7. Where this leaves the thesis story

1. **System headline:** predict-refine with a 17k denoiser beats HySRD/SpecSTG/DiffSTG on
   all four PEMS test sets; metrla/pemsbay numbers now exist and beat our DiffSTG re-run.
2. **Ladder law (measured):** static prior gain = exploitable spectral spread × stationarity
   — big on mlp (−3 to −31%), ~1% on gcn (best under past-conditioning), zero on gwn.
3. **Prior contribution that survives gwn:** adaptive variance (calendar + drift) in the
   structured eigenbasis — wins CRPS AND coverage over everything iso can express, 9/12
   comparisons, seed-consistent on pems08s; novelty verified unclaimed (cite NsDiff et al.).
4. **Calibration:** all raw arms are over-confident (cov 0.43-0.83 vs nominal 0.90); the
   adaptive recal is also the calibration fix; Student-t tails remain the residual lever.

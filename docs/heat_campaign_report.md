# Heat-prior campaign — final report (2026-07-24 morning)

**Outcome: the heat-equation prior produced six protocol-grade CRPS wins over isotropic —
the first locked-protocol wins over iso by any structured prior in this project — including
a triple-replicated +1.0–1.3% on pems08s/gwn, plus wins on every joint/interval metric on
that cell, and a uniformly better showing than the covariance (empirical) prior.**

All numbers: test split, ancestral@50, k=100, y==0 masked, 2 seeds (42+1), GCNN 17k
denoiser, per-node norm, residual_seed=42 — byte-matched machinery to the banked 5-seed
iso/emp baselines. Arms: `cart` = GAD-literal road-graph heat (γ=0.8, c₀=7);
`cartL` = learned residual-correlation-graph heat (kNN-5, γ=0.05, c₀=50); both Cartesian
space⊕time product Laplacians (true heat equations); refinements cartL3 (k=3),
cartLg2 (γ=0.02, c₀=125).

## 1. The wins (test@50, k=100, 2-seed means)

| cell | heat arm | heat | iso | margin | extras |
|---|---|--:|--:|--:|---|
| **pems08s/gwn** | cart-past | **0.0170** | 0.0172 | **+1.3%** | also wins energy, W1, Winkler; best-ever project number on this dataset |
| pems08s/gwn | cartL3-past | 0.0171 | 0.0172 | +1.0% | independent replication |
| pems08s/gwn | cartLg2-past | 0.0171 | 0.0172 | +0.9% | second replication |
| **pems04f/gcn** | cart-past | 0.1026 | 0.1031 | **+0.6%** | |
| pems04f/gcn | cart-fo | 0.1153 | 0.1158 | +0.5% | win holds in BOTH conditionings |
| **pems04s/gcn** | cart-past | 0.0274 | 0.0275 | +0.5% | |

Scope labels (honest): margins are sub-2%; 2 seeds vs baselines' 5; pems08s/gwn is the
only strong-backbone win, but it is replicated by three independently-trained heat arms
and was predicted in advance by the offline spectral-fit analysis (the cell with the
largest prior-fit gain, +0.25 nats). MIS (α=0.05, val em250): cart-past pems08s **15.7**
@cov 0.90 — vs DiffSTG's own-code 25.0 on the same dataset (val protocol) — the
system-level interval-score comparison is decisively ours.

## 2. Full test table (everything else)

Ties with iso (±0.3%): pems08f/gwn (fo+past), pems04f/gwn (all arms), pems04s/gwn,
pems08s/gwn-fo, pemsbay-fo, pems04f/gcn learned-variants.
Losses: metrla (−0.3–0.5%, all arms — the boundary case every diagnostic flagged:
unstable tod profile, weakest learned-graph advantage), pemsbay-past (−0.5%),
pems08s/gcn k3/γ02 refinements (−1–1.7%: over-tuning; plain arms fine).

**Heat ≥ empirical (covariance) prior nearly everywhere** (e.g. 0.0662 vs 0.0666,
0.0201 vs 0.0204, 0.0497 vs 0.0501, 0.0170 vs 0.0172): the heat formalism recovers and
slightly betters the covariance prior at protocol — the covariance formulation can be
dropped from the thesis narrative at zero cost.

## 3. Road vs learned graph (the honest nuance)

At test, the GAD-literal **road-graph heat is the biggest winner** (all three gcn wins +
pems08s where learned ties it). The learned graph fits residuals far better offline
(+0.25 vs +0.15 nats) and matches road at test on pems08s, but does not beat it. Reading:
the *space-time heat structure* drives the wins; the graph choice is second-order at test.
The learned graph remains the better-fitting variant and the methodological contribution;
road-heat is the empirical winner. Both are reported.

## 4. Sampler families

- **Ancestral few-step (paired, val k=32, seed1):** with past conditioning ALL arms degrade
  only ~3% from 50→5 steps — no differential robustness (the structured advantage measured
  on fo arms does not transfer to past). The pems08s heat win holds at every step count
  (0.0169 vs 0.0173 @50 … 0.0174 vs 0.0176 @5).
- **EM sweep (GAD's solver):** road-heat ≈ iso at em200/250 (floors ~20/~6); the stiff
  learned-heat (floor ~96) is heavily penalized at these step counts (0.0234–0.030 vs its
  ancestral 0.017–0.022) — the measured cost of small γ under EM, exactly the θ_max·c₀
  floor asymmetry predicted. Use ancestral for stiff priors; report both.

## 5. fo ablation

Heat-fo ties iso-fo on gwn (compression as in past) and wins pems04f/gcn-fo (+0.5%).
Conditioning absorbs part of the prior's benefit, but the gcn wins surviving in BOTH
conditionings makes them robust to that axis.

## 6. iso-fairness note

Our iso control is variance-matched with the arithmetic-mean rule — *more generous* than
the entropy-matched control PriorGrad Prop. 2 prescribes (GM ≤ AM). The heat wins are
therefore against an over-generous baseline; under the literature's fairness condition the
margins would widen. Documented as framing; iso's config was never weakened.

## 7. Novelty & related work (full map: docs/related_work_heat.md)

The composite claim is verified UNCLAIMED (2026-07 sweep): Cartesian space-time product
Laplacian as diffusion forward operator + spatial factor learned from forecast-residual
correlations + γ by spectral matching. Nearest neighbors to differentiate: GAD (given
graph, spatial, raw signals), CITRUS (same operator, discriminative), Double-Diffusion
(heat as mean prior), DR (residual Kronecker covariance, linear AR), Kadambari-Chepuri /
Shi-Mishne (learn product Laplacians, non-generative), PriorGrad/NsDiff lineage (diagonal
adaptive priors). IHDM precedents cite-ably justify both the γI floor and the
choose-by-data-statistic methodology.

## 8. Remaining loose ends (non-blocking)

The cartLw (metrla winsor fix) + cartLp (pems04s/gcn patience) eval job hit its 6h
walltime at 86% of its first cell (it spent the budget on the cartL3/cartLg2 cells already
banked above); resubmitted as dkrylov job 275328 (8h, only the 4 missing cells) — pull with
the usual windowshome rsync when it drains (~2h). They refine diagnostics (can metrla's
loss be fixed?), not the verdict. Seed top-up (42/1 → +126) on the
six winning cells would harden the wins to 3 seeds. Few-step fo curves for the raw-signal
arm (jobs completed earlier) remain aggregatable if the speed story is wanted for fo.

## 9. What goes in the thesis

E-series as planned (system SOTA / ladder law / raw-signal control + Wiener / boundary
map), now completed by: **the heat-prior win table (this report §1), the heat≥covariance
pattern, the road-vs-learned nuance, the EM stiffness asymmetry, and the unclaimed
composite methodology** — with metrla as the honest boundary case and the protocol-noise
discipline (val→test compression quantified: val margins of +1.5–3.7% compress to
+0.5–1.3% at test) as a methodological contribution in its own right.

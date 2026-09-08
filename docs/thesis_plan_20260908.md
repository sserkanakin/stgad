# Thesis plan — the story, chapter by chapter (2026-09-08)

One sentence: *a diffusion model for traffic-forecast errors whose noise already has the shape of
those errors — correlated along the horizon, along the road graph, and network-wide — forecasts
each sensor as well as white noise does, and the network-level uncertainty correctly, which
white noise cannot do by construction.*

Status legend: **have** = measured on the fresh pipeline · **running** · **to do** (CPU on the
Mac unless marked GPU). Configuration detail always goes to Appendix B.

---

## Front matter
Summary (½ page): the sentence above + the three numbers (per-point tie; network-total
coverage 0.64 → 0.72 → 0.9 with the DC rule; 6/6 datasets). Nomenclature from the symbol table
in `docs/method_walkthrough_20260907.md` §10.

## 1 Introduction (rewrite, ~4 pages)
Story: a good forecaster is wrong by a structured amount; operators need the uncertainty of
*aggregates* (corridor travel time, network flow), not only of single sensors; existing
diffusion forecasters use white noise and are correct per point yet wrong on aggregates by
2–4×. Research questions (keep three):
- RQ1 What structure does the residual field have, and can a graph-based Gaussian prior capture it? (Ch. 5)
- RQ2 Does diffusing with that prior instead of white noise change what the forecaster gets right? (Ch. 6)
- RQ3 Where does the remaining error come from — prior, sampler, or network — and what fixes it? (Ch. 6–7)
Contributions: (i) the matched Kronecker prior with a global edge; (ii) the exact per-mode sampler
and the derived floor/step rule; (iii) the joint evaluation and the 6/6 result; (iv) the
diagnosis that the learned graph filter fails on the DC mode and the prior's filter for it.
**to do** (text only).

## 2 Background (keep, trim)
Keep the existing sections; cut anything about Student-t, reshaping, product graphs. Add one
paragraph each on CRPS_sum / PICP / MSIS (standard names, citations) so Ch. 6 needs no
definitions. **to do** (edit).

## 3 Literature review (keep, update positioning)
Positioning sentence: GAD (graph-aware forward, unconditional signals, aMMD + steps curve) →
we extend to conditional spatio-temporal residuals with a Kronecker prior, a global edge, an
exact sampler, and forecasting metrics; DiffSTG/SpecSTG/HySRD report per-point CRPS only and
never CRPS_sum. **to do** (edit).

## 4 Methodology (keep structure, update content)
4.1 Backbone and residual field (as is).
4.2 The structured prior — Σ_T from the residual; Σ_S = unit-diagonal (L_β + γI)⁻¹ with the
    global edge; the two-moment interpretation of (β, γ); the matching rule; θ and the
    scale normalisation. Figures: the 2×2 Kronecker toy matrix; Σ_S correlation maps
    (road-only vs matched). **have** (walkthrough §4).
4.3 Structured diffusion — forward OU, FCPS schedule, log-uniform noise levels, SNR loss,
    the GCNN denoiser and its conditioning [X_z | Ŷ_z], the exact ancestral step *with* the
    posterior-variance term, the derived floor f(c_lo) = 1.3e-3 and 16 steps/decade, and the
    DC rule (prior's filter for the network-wide mode). **have** (walkthrough §5–7).
4.4 Evaluation protocol — per point (CRPS, MAE, PICP, MSIS), joint (CRPS_sum on the network
    total, cumulative per sensor, spread ratio), GAD's aMMD. **have**.
4.5 Cost (params, steps, seconds per window). **have** from metrics.
Conclusion paragraph.

## 5 The prior and the residual field (NEW analysis chapter — `docs/prior_analysis_plan_20260907.md`)
The chapter that answers RQ1 with measurements, one section per question, each with a
figure, the claim, and the objection answered.
5.1 The residual field — horizon growth + R_H; correlation vs hops; common-mode share; the
    network-total sd vs independence; tails; train→test drift. **have** (pems08f/s, pems04f), **running** (rest).
5.2 How close is the prior — graph-spectral variance vs λ (data vs iso / road-only / matched);
    mode rank agreement; 91–94 % separability; **the multi-scale figure** (sensor → hop →
    clusters → network: matched ≈1 everywhere, iso and road-only collapse at the network);
    held-out NLL; the neighbour over-correlation as the honest cost; (β, γ) surface. **have/running**.
5.3 What the prior changes — prior draws on the map; derived quantities (total, neighbour
    difference, horizon increment); the θ spectrum. **have** (numbers), maps **to do**.
5.4 The forward process GAD-style — frequencies going down (energy per λ band vs c̄, iso vs
    kron); convergence to the prior; f(c̄) with the old 0.02 floor vs the derived ones. **to do** (CPU, 1 day).
Conclusion: the prior is right at every scale; its one systematic error (neighbour
correlation) and one systematic risk (test drift) are named.

## 6 Experiments (rebuild around the fresh campaign)
6.1 Setup — datasets, backbones, the two arms, seeds, protocol (short; detail → App. B). **have**.
6.2 Per-point results — SOTA table: published / GWN+Gaussian / iso / kron. Claim: tie between
    arms; both below every published number; the backbone owns this metric. **have**.
6.3 Joint results — the main table and figure: CRPS_sum, PICP, spread ratio on the network
    total and the cumulative, iso vs kron, 6 datasets, seed points. Claim: kron better on every
    joint metric on 6/6 datasets. **have**.
6.4 Where the rest of the uncertainty went — the chain diagnostic (learned / oracle / hybrid),
    the DC rule applied to all cells: total coverage → ≈0.9, per-point unchanged or better,
    iso unchanged. Claim: sampler exact; network fails on one mode; prior fixes it. **running (GPU)**.
6.5 The role of the backbone — MLP / GCN / GWN ladder: the prior's joint gain vs backbone
    strength (and per-point gain for the weak backbones). **to do (GPU, ~17 h, two waves)**.
6.6 GAD-style evaluation — aMMD (QV / SC / DC) of samples vs test residuals; quality vs number
    of reverse steps for both arms (and EM's stability count). **to do (eval-only GPU + CPU)**.
6.7 Ablations — old sampler vs corrected; fixed floor vs derived; road-only γ=0.8 and
    temporal-only priors at prior level (and at sample level on two datasets if time). **to do**.
6.8 Calibration figures — reliability and PIT per point and for the total. **have** (diag npz).
Conclusion paragraph.

## 7 Discussion
7.1 The mechanism — why per-point cannot see the prior and the total must; the DC-mode
    finding as the bridge between prior and network; what the loss weighting implies.
7.2 Against the old results — the same comparison under the old sampler said the opposite;
    what each fix changed (table from the archived vs fresh comparison).
7.3 Limitations — Gaussian prior on heavy-tailed residuals (denoiser supplies tails), test
    drift (pems08s 2.4×), neighbour over-correlation, one backbone family, matching uses one
    moment.
7.4 Implications for practice — network-level intervals for operations; cost.
**to do** (text; all inputs measured).

## 8 Conclusion
Answers to RQ1–3; contributions; future work: mode-aware loss weight, global channel in the
denoiser, two-moment match, adaptive scale. **to do** (text).

## Appendices
A Forward-process derivation (keep). B Configuration and derived per-cell numbers (the table
from the walkthrough), full seed grids, re-evaluation notes, run times. C The (β, γ) grid.

---

## Order of work from here
1. GPU: finish the DC-rule pass (running) → start the MLP wave → GCN wave → eval-only pieces
   (steps curve, sampler ablation, sample dumps for aMMD).
2. CPU: prior-chapter figures (§5.1–5.4), then the sample-level figures from the dumps.
3. Text: Ch. 6 tables/figures from `aggregate_matched.py` + `results/analysis/*.json`; Ch. 5 from
   `results/analysis/prior/*.json`; then Ch. 1, 3, 7, 8 edits; App. B last.

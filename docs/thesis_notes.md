# Thesis notes — decisions, one step at a time

*Running log of every discussion about the thesis: what we decided to say, which figure carries it,
what we leave out, and why. One step per conversation; the step is closed only when the user says so.
Method reference: `docs/method.md`. Results: `results/analysis/*.json`, figures `results/figures/`.*

## How we work through it

| step | topic | outcome we need before moving on |
|---|---|---|
| 0 | scope and claim | one paragraph everyone agrees with: problem, claim, what we do NOT claim, the research questions |
| 1 | Ch. 1 Introduction | motivation story, RQs, contribution list, outline |
| 2 | Ch. 2 Background | the concepts the reader needs and at what depth (Laplacian/GSP, Kronecker, diffusion, GAD, the metrics) |
| 3 | Ch. 3 Literature | positioning: what others do, the gap, one sentence per neighbour paper |
| 4 | Ch. 4 Methodology | section list, formulas kept in the body vs appendix, the two figures (toy Kronecker; Σ_S maps) |
| 5a | Ch. 5.1–5.2 setup + residual field + prior | figures 1–6, the claims and the objections answered |
| 5b | Ch. 5.3 main result | the joint figure/table, the per-point tie, how to phrase the win |
| 5c | Ch. 5.4 ladder | MLP/GCN/GWN figure, the "when does structure help" sentence |
| 5d | Ch. 5.5–5.7 baselines, calibration, ablations | SOTA table, old-vs-new, chain diagnostic, DC rule, two-moment prior |
| 6 | Ch. 6 Discussion | mechanism, limitations, practice |
| 7 | Ch. 7 Conclusion + appendices | answers to RQs, future work, App. B contents |

For each step: we talk → decisions are written here → the LaTeX section is drafted from the decisions →
the user reads it → the step is closed.

---

## Step 0 — scope and claim (open)

**Proposed scope, in plain words (to be corrected by the user):**

A strong deterministic traffic forecaster (Graph WaveNet) is wrong by a structured amount. We model
the distribution of that error with a diffusion model. The one thing we change, and the thesis is
about, is the *noise* the diffusion model corrupts with and starts from: instead of white noise we use
a Gaussian whose covariance is built from the road graph (heat kernel + one weak global edge) and the
error's own horizon structure (Kronecker product), with its two free scalars set so the prior's
uncertainty of the network-wide total matches the data. Everything else — data, backbone, denoiser
(17k parameters), training, sampler, evaluation — is identical between the white-noise arm and ours.

**What we claim (measured):**
1. The residual field is structured (neighbour correlation, horizon correlation, a network-wide
   component); a Kronecker graph prior captures it at every aggregation scale; white noise and the
   road-only prior do not (they collapse at the network scale).
2. With the prior, per-sensor accuracy is unchanged (a tie; both arms below published methods) and
   every network-level uncertainty measure improves on all six datasets (CRPS_sum −3…−6 %, coverage
   of the total +6–8 points, spread +21–53 %), 4 seeds.
3. The weaker the backbone, the larger the gain — with an MLP backbone the prior also wins per
   sensor (−1.5…−6 % CRPS) and by −4…−24 % CRPS_sum.
4. The remaining under-dispersion of the total is the denoiser's treatment of one mode (the
   network-wide shift); the sampler is exact; the prior's filter for that mode restores coverage
   (an ablation, not the final fix).

**What we do NOT claim:** better point accuracy than the backbone; a learned prior; heavy-tailed
modelling; that the prior is right under distribution shift (PEMS08 speed / PEMS-BAY test drift).

**Research questions (draft):**
- RQ1 What structure does the forecast residual have, and can a graph-based Gaussian prior capture it?
- RQ2 Does diffusing with that prior, instead of white noise, change what the forecaster gets right?
- RQ3 Where does the remaining error come from — prior, sampler or network — and what fixes it?

**Decisions:** *(none yet — waiting for the user)*

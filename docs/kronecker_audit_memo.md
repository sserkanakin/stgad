# Kronecker prior audit — is the thing the drift multiplies by correct?

Date: 2026-07-26. Scope: audit the Σ = Σ_T ⊗ Σ_S diffusion prior end-to-end — the code
that applies it, and whether it is the right *target* for the residual. Offline, on saved
residuals (`results/residuals/<ds>/<bb>/seed42`), test split held out.

Reproduce:
- mechanics: `.venv/bin/python scripts/analysis/verify_process.py --dataset <ds> --backbone gwn --temporal empirical`
- modelling: `.venv/bin/python scripts/analysis/kronecker_audit.py --split test`
  (→ `results/analysis/paper_results/kronecker_audit.md` + `figs/kron_audit_whitening_*.png`)

## The setup, stated precisely

The forward SDE is `dr = −c(t)·Q·r dt + √(2cσ²) dW`, `dW` white, `Q = Σ⁻¹ = Σ_T⁻¹ ⊗ Σ_S⁻¹`.
**The injected noise is isotropic; all Kronecker structure lives in the drift operator `Q`.**
In the joint eigenbasis `V_S ⊗ V_T` the drift pushes each graph×time mode toward zero at
rate `θ[k,h] = q_S[k]/(τ²·μ_T[h])`. So "is the thing we multiply by correct" = "is `θ` the
residual's true per-mode precision" — a whitening question.

Two factors, built very differently:
- **Σ_T (temporal)** is **fit** to the residual — `D_H` (per-horizon variance) × `R_H`
  (empirical horizon correlation). Matches the residual's temporal covariance by construction.
- **Σ_S (spatial)** is **imposed** — `(L+γI)⁻¹`, road graph, γ=0.8 fixed, **never fit to the
  residual.** This is the arm that can be flat-out wrong.

## Verdict 1 — the code applies Σ_T⊗Σ_S faithfully ✅

Read `prior.py`, `forward_fcps.py`, `reverse.py`. The 2-D rotation `C = V_Sᵀ R V_T` **is**
the Kronecker eigenbasis rotation `(V_T⊗V_S)ᵀ vec(R)` (V_S on nodes, V_T on horizons); the
index convention is self-consistent; per-mode precision `θ = q_S ⊗ (1/μ_T)` is the correct
outer product; the forward marginal `H=e^{-c̄θ}`, `σ²_t=σ²(1−e^{−2c̄θ})/θ` is the exact
per-mode OU, and both reverse samplers invert that same algebra. There is even a guard
comment (`forward_fcps.py:453`) about a past index bug that would have scrambled the
temporal covariance — already caught. `verify_process.py` confirms with numbers:

| dataset | [A] stat.var | [B] forward→stat | [C] mix 2θ_min·c₀ | [D] heat order |
|---|--:|--:|--:|:--|
| metrla/gwn | 0.018 | 0.020 | 11.2 (≥5) | PASS |
| pems08f/gwn | 0.019 | 0.021 | 11.2 (≥5) | PASS |

**The "space scrambled with time" class of bug is absent.**

## Verdict 2 — the *form* Σ_T⊗Σ_S is right: the residual covariance IS separable ✅

Van Loan–Pitsianis separability index (fraction of `Cov(vec r)` energy any single Σ_T⊗Σ_S
can represent), top-1 across all 18 cells: **0.84–0.99** (top-3: **0.94–1.00**). MLP residuals
are the most separable (~0.99, no space–time mixing); GWN leaves the least (0.84–0.97, it
mixes space and time). **A Kronecker product is the right shape of model everywhere.**

## Verdict 3 — the temporal factor carries ~everything; the imposed graph factor is near-inert ⚠️

Held-out (test) per-mode NLL of a nested family sharing one total variance, differing only
in shape. ΔT = what **time** buys over iso, ΔS = what the **graph** buys over iso, `graph|T`
= what the graph adds **on top of** the temporal factor (nats/mode, positive = better):

| ds/gwn | iso | T-only | S-only | full | **ΔT** | **ΔS** | **graph\|T** | oct_full |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| metrla  | 0.567 | 0.198 | 0.506 | 0.170 | **0.369** | 0.062 | **+0.028** | 0.57 |
| pemsbay | 0.735 | −0.032 | 0.646 | −0.079 | **0.767** | 0.089 | **+0.047** | 1.02 |
| pems04f | 0.542 | 0.433 | 0.546 | 0.453 | **0.109** | −0.004 | **−0.020** | 0.90 |
| pems04s | 0.742 | 0.041 | 0.757 | 0.069 | **0.701** | −0.015 | **−0.028** | 0.95 |
| pems08f | 0.500 | 0.343 | 0.486 | 0.344 | **0.157** | 0.014 | **−0.000** | 0.59 |
| pems08s | 0.834 | 0.014 | 0.824 | 0.027 | **0.819** | 0.009 | **−0.013** | 1.02 |

(mlp/gcn: ΔT even larger, up to **1.43**; `graph|T` still ≈0.) The pattern is uniform:

- **Time buys 0.1–1.4 nats/mode — the whole value of the prior.** Σ_T is fit, so it correctly
  re-times the drift to the residual's strong horizon structure.
- **The road-graph spatial factor adds ≈0 on top of time, and on flow it slightly HURTS**
  (`graph|T` negative for all four pems-flow/speed cells on gwn). The imposed graph smoothness
  is not the residual's spatial organisation.

### Whitening maps make the spatial mismatch visible
`figs/kron_audit_whitening_*.png`. Median |octaves off| of the full prior: metrla **0.29**,
pems08f **0.33**, pems04f **0.64**. The temporal marginal (prior vs held-out) matches tightly
everywhere; the spatial marginal shows held-out energy scattered around — not concentrated on —
the smooth graph modes the prior bets on. For pems04f the (N,H) whitening map has a systematic
**blue band (smooth graph modes over-crushed) over a red band (rough modes under-crushed)**:
the drift dampens the wrong spatial directions.

### Important nuance — the residual IS low-frequency; the strong backbone already ate it
The spatial factor is small on gwn **not** because the residual is spatially white — it isn't.
Graph-spectral energy of the residual (Dirichlet energy, white=1; lower=smoother):

| ds/bb | Dirichlet | DC-mode share | smooth-10% (white=.10) | corr w/ 1/(λ+γ) |
|---|--:|--:|--:|--:|
| metrla/mlp | 0.455 | **0.269** | 0.551 | 0.476 |
| metrla/gcn | 0.744 | 0.064 | 0.280 | 0.697 |
| metrla/gwn | 0.756 | 0.041 | 0.264 | 0.790 |
| pems04f/mlp | 0.545 | 0.120 | 0.391 | 0.339 |
| pems04f/gwn | 0.873 | 0.023 | 0.153 | 0.342 |
| pems08f/gwn | 0.841 | 0.036 | 0.162 | 0.546 |

Every residual is smoother than white (low-frequency). But **MLP** (no graph model) leaves it
STRONG — 27% of spatial energy in the single DC/global-average mode — while **GWN's graph
convolutions absorb most of it** (DC-mode 2–4%, Dirichlet ~0.76–0.87). The graph prior's benefit
tracks `(low-frequency-ness) × (alignment with 1/(λ+γ))`, exactly matching ΔS across backbones
(metrla ΔS: mlp 0.169 → gcn 0.064 → gwn 0.062) and the one positive gwn cell being the most
low-freq + best-aligned (metrla/gwn, Dirichlet 0.756, corr 0.79). So: **the residual is
low-frequency, but the strong spatio-temporal backbone has already exploited that structure —
leaving the diffusion prior's spatial half little to add on top of the (fit) temporal factor.**
This is the "backbone is a confounder" point: the spatial prior would matter more paired with a
backbone that doesn't model space (mlp/gcn), or on the raw signal (GAD's setting), than on a GWN residual.

### The spatial factor is not intrinsically dead — it is the wrong *graph*
`graph|T` with the road graph vs a residual-correlation kNN graph:

| ds/gwn | road | correlation |
|---|--:|--:|
| pems04f (flow) | −0.020 | **+0.011** |
| pems08f (flow) | −0.000 | **+0.026** |
| metrla (speed) | **+0.028** | +0.013 |

Flow wants the correlation graph (volume correlates by function, not road distance); speed
wants the road graph (congestion propagates along roads). But even matched, the spatial gain
(~0.01–0.03) stays an **order of magnitude below temporal** (~0.1–1.4).

## Why THIS Kronecker (matrix-normal) vs the GTCNN/GAD product-graph
Two different objects. **Ours**: Σ = Σ_T ⊗ Σ_S = Kronecker product of two *covariances* — a
**matrix-normal** / separable GP. Precision is **multiplicative** θ=q_S·q_T. **Product-graph
(GTCNN/GAD Cartesian, our `temporal="cartesian"` arm)**: Q = L_◇+γI on the Cartesian product
graph; precision is **additive** θ=λ_S+λ_T+γ, bounded [γ,2+γ]. Head-to-head held-out per-mode
NLL (gwn, lower=better):

| ds/gwn | empirical Σ_T⊗Σ_S (NLL / oct / stiff) | cartesian product-graph (NLL / oct / stiff) |
|---|--:|--:|
| metrla | **0.170** / 0.57 / 52 | 0.320 / 0.98 / 20 |
| pemsbay | **−0.079** / 1.02 / 139 | 0.373 / 1.81 / 20 |
| pems04f | **0.453** / 0.90 / 18 | 0.484 / 0.98 / 21 |
| pems08f | **0.344** / 0.59 / 22 | 0.395 / 0.78 / 21 |

Empirical wins everywhere. Reason: (1) a separable residual covariance (VLP 0.84–0.99) IS a
Kronecker product of covariances; the Cartesian `(L_◇+γI)⁻¹` is the inverse of a *sum* and
cannot represent a separable covariance exactly — it is the mis-specified family here. (2) The
multiplicative form is the only one that lets Σ_T be the **empirical** horizon covariance (the
Cartesian forces a path-graph temporal smoothness model), and ΔT shows that is where all the
value is. Cost: multiplicative squares the dynamic range → stiff prior (needs FCPS + more
sampler steps); the Cartesian is bounded/gentler but fits worse. NB roles differ: the GTCNN
product graph is a *convolution operator*; our Kronecker is the *noise covariance* — orthogonal
choices (a GTCNN denoiser could sit on top of this matrix-normal prior).

## Is the temporal drift a "heat equation"? No — and that's fine
"Heat equation" ⟺ the drift operator is a graph **Laplacian** (smoothing). Our temporal
`Q_T = Σ_T⁻¹` is the inverse of a *fitted covariance*, not a Laplacian — so the temporal drift
is a **variance-matched Ornstein–Uhlenbeck** (drain low-variance temporal modes first, protect
high-variance), NOT heat flow along the horizon. Only the spatial `(L_S+γI)⁻¹` factor (and the
`cartesian` arm) is a genuine heat equation. This is legitimate: *any* `N(0,Σ)` is the stationary
law of `dr=−cΣ⁻¹r dt+√(2cσ²)dW` (VP=`Σ⁻¹=I`, GAD-heat=`L+γI`, ours=`Σ_T⁻¹⊗Σ_S⁻¹` — all one
family). The forward's objective is to match the data covariance, not to diffuse heat; since the
residual is temporally structured but NOT path-smooth, the empirical `Σ_T` is *more* correct than
a temporal heat equation (ΔT and the empirical-vs-cartesian table confirm).

**Honest tension (the real result):** matching the *prior* ≠ better *samples*. Baking `Σ_T` into
the forward is a design choice; an isotropic forward + a capable denoiser can learn the same
structure. Our own end-to-end numbers: **iso ties/beats the structured Kronecker on gwn test CRPS**
(see [[project_test_verdict_20260722]]). So the structured forward improves the terminal-prior fit
(ΔT large) but does not translate to better generation on the strong backbone — and it costs a
*fitted* operator (train→test drift; the whitening octaves) and *stiffness* (52/139 vs the Laplacian's
~20 → FCPS + many sampler steps). Frame the prior as a "separable Gaussian forward (graph-heat spatial
factor × empirical temporal factor)", not a "graph-time heat equation"; the scientific finding is that
this prior-matching does not by itself beat an isotropic forward end-to-end once the backbone is strong.

## Sampler fidelity note (not a bug)
`verify_process.py [E]` with an **oracle Wiener denoiser** (isolates sampler from net): the
exact-posterior **ancestral** sampler recovers the target second moment at ~50–100 steps on
mild priors (pems08f stiffness 22) but needs more on stiffer ones (metrla stiffness 52 still
2.4× the batch-floor at n=100). **EM** needs `n ≳ θ_max·c₀` just for stability and more for
accuracy. Production eval defaults to EM `n_steps=1000` (auto-floored to θ_max·c₀), well above
both — so this is a *fidelity-scales-with-stiffness* note, and a reason to prefer the ancestral
sampler, not a correctness defect.

## Bottom line
1. **Code**: correct. The Kronecker is applied on the right axes and the forward reaches
   `N(0, Σ_T⊗Σ_S)`.
2. **Form**: correct. The residual covariance is 84–99% Kronecker-separable.
3. **Content**: the contribution is overwhelmingly the **empirical temporal factor** Σ_T. The
   spatial graph factor is a small correction that is only positive when the graph matches the
   residual (road↔speed, correlation↔flow), and is inert/harmful otherwise.

**Implication for the write-up:** frame the prior honestly as an *empirical temporal
re-timing* of the diffusion, with a *conditional* spatial term ("structure helps when the
graph matches the residual — road for speed, correlation for flow"). Do not sell a
"spatio-temporal graph prior" whose spatial half is near-inert on the production (gwn) arm.
The rigorous "when does structure help" story is stronger and is what the numbers support.

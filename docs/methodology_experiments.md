# Methodology and Experimental Design — Graph-Aware Spatio-Temporal Diffusion (GAD-ST)

*Thesis chapters 4 (Method) and 5 (Experimental Design). Branch `raw-signal-gad`.
Results tables are scaffolded and filled from the live experiments
`20260716_gadst` (unconditional) and `20260716_gadst_cond` (conditional).*

---

## 4. Method

### 4.1 Problem setup

We are given a sensor network described by an undirected weighted graph
$\mathcal{G}_S=(\mathcal V,\mathcal E,\mathbf A)$ with $N=|\mathcal V|$ sensors and a
symmetric adjacency $\mathbf A\in\mathbb R^{N\times N}$ built from the road network.
At each time step every sensor emits a scalar reading (traffic speed or flow); over a
window of $H$ future steps the network state is a matrix
$\mathbf X\in\mathbb R^{N\times H}$. We study probabilistic forecasting: rather than a
point prediction we model a distribution over the future window, from which we can draw
samples and quantify uncertainty.

We attack the problem in two framings, which correspond to the two experiments of
Chapter 5:

- **Unconditional generation** (GAD's setting, lifted one axis up). Learn the
  distribution $p(\mathbf X)$ of full space–time windows and generate new ones. This
  isolates the generative machinery from the forecasting signal and is where GAD's own
  advantage is claimed.
- **Conditional forecasting** (predict-and-refine). A deterministic backbone
  (Graph-WaveNet, GWN) produces a point forecast $\hat{\mathbf Y}$; we learn the
  distribution of the *residual* $\mathbf R=\mathbf Y-\hat{\mathbf Y}$ conditioned on the
  forecast, and refine $\hat{\mathbf Y}+\mathbf R$ into a calibrated predictive
  distribution. This is the deployable forecaster.

Both framings use the *same* diffusion prior and denoiser, so the pair answers a single
question from two sides: **where, if anywhere, does graph structure in the corruption
process help?**

### 4.2 Background: score-based diffusion

A score-based generative model defines a forward stochastic differential equation (SDE)
that gradually turns data into noise,
$$\mathrm d\mathbf x = \mathbf f(\mathbf x,t)\,\mathrm dt + g(t)\,\mathrm d\mathbf w,$$
and generates by integrating the corresponding reverse-time SDE, which depends on the
score $\nabla_{\mathbf x}\log p_t(\mathbf x)$. Two standard choices frame our baselines:

- **VP-SDE** (variance preserving; the SDE limit of DDPM): $\mathrm d\mathbf x=
  -\tfrac12\beta(t)\mathbf x\,\mathrm dt+\sqrt{\beta(t)}\,\mathrm d\mathbf w$, isotropic,
  stationary $\mathcal N(\mathbf 0,\mathbf I)$.
- **VE-SDE** (variance exploding): $\mathrm d\mathbf x=\sqrt{\tfrac{\mathrm d\sigma^2}{\mathrm dt}}\,\mathrm d\mathbf w$,
  isotropic, terminal $\mathcal N(\mathbf 0,\sigma_{\max}^2\mathbf I)$.

In both, the noise geometry is isotropic: every coordinate is corrupted independently and
equally. The network is trained to denoise, and the graph — if used at all — enters only
through the denoiser architecture, never through the noise.

### 4.3 Graph-Aware Diffusion (GAD)

GAD (Rozada et al., *Graph-Aware Diffusion for Signal Generation*, arXiv:2510.05036;
Isufi co-author) replaces the isotropic forward process with the **graph heat equation**
$$\mathrm d\mathbf x = -c(t)\,(\mathbf L+\gamma\mathbf I)\,\mathbf x\,\mathrm dt
   + \sqrt{2c(t)}\,\sigma\,\mathrm d\mathbf w,$$
where $\mathbf L$ is the normalized symmetric graph Laplacian and $\gamma>0$ a small shift
that makes $\mathbf L+\gamma\mathbf I$ positive definite. Diffusing along the graph rather
than isotropically has two consequences GAD exploits:

1. **Structured stationary law.** As $t\to\infty$ the process converges to a Gaussian
   Markov random field (GMRF),
   $\mathbf x_\infty\sim\mathcal N\!\big(\mathbf 0,\ \sigma^2(\mathbf L+\gamma\mathbf I)^{-1}\big)$,
   whose covariance is parametrized by the graph. Smooth (low graph-frequency) modes carry
   more variance; the terminal noise is already graph-smooth rather than white.
2. **Graded corruption.** In the Laplacian eigenbasis each mode $k$ is an
   Ornstein–Uhlenbeck process with rate $c(t)\,\theta_k$, $\theta_k=\lambda_k+\gamma$.
   High-frequency modes (large $\lambda_k$) decay fast; low-frequency modes are preserved
   until late. GAD schedules $c(t)$ with the **floor-constrained polynomial schedule
   (FCPS)** so slow modes still mix by $t=T$.

The reverse process is a sequence of graph-signal denoising problems, which motivates a
**graph convolutional denoiser** (a cascade of graph filters, GAD Eq. 16). GAD reports
that this graph-aware process beats isotropic VP/VE diffusion **especially at small numbers
of reverse steps** — the regime where the structured prior gives the sampler a head start.

### 4.4 GAD-ST: the spatio-temporal extension (contribution)

GAD generates *spatial* graph signals (one value per node). Forecasting requires *space–time*
windows $\mathbf X\in\mathbb R^{N\times H}$. Our contribution is to carry the graph heat
process onto a **product graph** over the $N\!\cdot\!H$ space–time vertices.

**Product graph.** We form the normalized **Cartesian product**
$\mathcal G_{ST}=\mathcal G_{\text{road}}\ \square\ \mathcal P_H$, where $\mathcal P_H$ is a
path graph on the $H$ forecast horizons (temporal chain). Its Laplacian has the
Kronecker-structured eigenbasis
$$\mathbf V_{ST}=\mathbf V_S\otimes\mathbf V_T,\qquad
  \lambda_{ST}[k,h]=\tfrac12\big(\lambda_S[k]+\lambda_T[h]\big)\in[0,2].$$
The eigenvalues **add** (Cartesian product), so the joint spectrum stays bounded in $[0,2]$
exactly as a normalized Laplacian should. We deliberately reject two alternatives:

- the **Kronecker (tensor) product graph**, whose eigenvalues *multiply* — it folds
  frequencies (a rough spatial mode times a rough temporal mode becomes a smooth joint
  mode), which measured *worse* than isotropic on most datasets;
- the **strong product**, which mixes both and inherits the folding.

A parametric product-graph sweep (Mazzola MSc thesis, TU Delft 2020, sup. Isufi, Eq.
2.24–2.25) confirms the Cartesian weight is selected empirically (learned tensor weight
$s_{11}\!\approx\!0$).

**Prior.** The GAD-ST precision is $\mathbf Q=\mathbf L_{ST}+\gamma\mathbf I$ with a single
fixed shift $\gamma=0.8$ (GAD's value; see 5.3), giving per-mode precision
$\theta[k,h]=\lambda_{ST}[k,h]+\gamma\in[0.8,2.8]$ and stationary GMRF
$\mathcal N(\mathbf 0,\sigma^2\mathbf Q^{-1})$. The scalar $\sigma^2$ is matched once to the
standardized-data variance. **Two scalars total** ($\gamma,\sigma^2$); no fitted covariance.
The precision spread (stiffness) is $(2+\gamma)/\gamma=3.5$ — GAD's own gentle regime, so
the forward is a graded cascade rather than an early collapse.

**Forward process** is the product-graph heat equation with the same FCPS schedule. In the
joint eigenbasis every mode is an independent OU process, so the marginal $p(\mathbf x_t\mid
\mathbf x_0)$ is closed-form:
$$\mathbf x_t = e^{-\theta\bar c(t)}\,\mathbf x_0
   + \sqrt{\sigma^2\,\tfrac{1-e^{-2\theta\bar c(t)}}{\theta}}\;\boldsymbol\epsilon,
   \qquad \bar c(t)=\textstyle\int_0^t c(s)\,\mathrm ds,$$
which recovers the stationary variance $\sigma^2/\theta$ as $\bar c\to\infty$. Training
needs only this closed form — it never simulates the SDE — so it is sampler-agnostic.

**Normalization.** The residual's standard deviation grows $1.7$–$2.9\times$ across the
$H$ horizons, whereas the stationary prior is flat in time. We standardize per horizon
(12 scalars from train statistics) so the target matches the prior; the growth re-enters
the samples at de-normalization. Both the graph and isotropic arms use identical
normalization, so it cannot bias the comparison.

**Reverse process.** We use two samplers:

- **Ancestral** — an exact per-mode exponential integrator of the reverse OU bridge; has
  no stability floor and is exact between grid points ($\approx$ 5–50 denoiser calls).
- **Euler–Maruyama (EM)** — GAD's own sampler, included for parity; its reverse drift is
  GAD Eq. 6 with the Tweedie score (Eq. 7). EM has a stability floor of $\theta_{\max}\!\cdot\!S$;
  at $\gamma=0.8$ this is $\approx 19$ steps, so every step count we report ($\ge 20$) is
  fully resolved.

**Denoiser.** All arms share GAD's graph convolutional denoiser: a cascade of three graph
filters ($K=5$ taps each over the shift operator $\mathbf S=$ normalized Laplacian), hidden
width $C=24$, SiLU activations, LayerNorm, residual output $\mathbf x_0=\mathbf x_t+
\text{net}(\mathbf x_t,t)$ ($\approx$ 17k parameters). Diffusion time is injected by a
sinusoidal embedding concatenated to the node signal. In the conditional arm the forecast
is supplied as additional conditioning channels.

**Loss.** GAD's signal-to-noise-weighted $\mathbf x_0$ regression plus a graph
total-variation regularizer on the prediction,
$$\ell = \mathbb E\big[w(t)\,\|\mathbf x_0-\hat{\mathbf x}_0\|^2\big] + 5\!\cdot\!10^{-3}\,
   \hat{\mathbf x}_0^\top\mathbf L_{\text{comb}}\hat{\mathbf x}_0,
   \qquad w(t)=\tfrac{1}{1+2\sigma^2 c(t)},$$
masked at missing sensors. The TV term (GAD's `tv_pred`, combinatorial Laplacian) nudges
predictions toward graph-smoothness.

### 4.5 The controlled comparison (four arms)

The experiment holds the denoiser, training, normalization, loss, and evaluation fixed and
varies **only the forward SDE / prior**. This isolates the effect of noise geometry:

| arm | forward process | stationary law | role |
|---|---|---|---|
| **GAD-ST** (`kron`) | product-graph heat, $\mathbf Q=\mathbf L_{ST}+0.8\mathbf I$ | graph GMRF | the method |
| **VPD** | VP-SDE (DDPM), $\theta=1$ | $\mathcal N(\mathbf 0,\mathbf I)$ | GAD's paper baseline |
| **VED** | VE-SDE, $\theta\to0$, geometric $\sigma$ | $\mathcal N(\mathbf 0,\sigma_{\max}^2\mathbf I)$ | GAD's paper baseline |
| **ISO** | isotropic heat, $\sigma^2\mathbf I$, heat-matched | $\mathcal N(\mathbf 0,\sigma^2\mathbf I)$ | **the control GAD omits** |

VPD and VED reproduce GAD's own adversaries natively in our codebase (no external code).
The **isotropic heat control** is the methodological point: it has the *same total heat and
schedule* as GAD-ST but white noise geometry, so any GAD-ST advantage over it is
attributable to the graph and nothing else. GAD's paper compares only against VPD/VED — a
different schedule *and* different geometry at once — and so cannot separate the two. Our
four-arm design can.

---

## 5. Experimental Design

### 5.1 Research questions

- **RQ1 (generation).** Does the graph prior improve unconditional space–time generation,
  and is the advantage concentrated at small step counts as GAD claims?
- **RQ2 (forecasting).** Does the graph prior improve conditional forecasting, or does the
  conditioning signal bypass the corruption prior?
- **RQ3 (attribution).** Of any GAD-ST advantage over VPD/VED, how much is noise *geometry*
  (vs the matched-isotropic control) and how much is *schedule*?

### 5.2 Datasets

Six standard traffic benchmarks; METR-LA and PEMS08 are the headline pair (most-benchmarked
speed/flow), PEMS-BAY and PEMS04 extend the sweep.

| dataset | modality | $N$ | horizon $H$ | provenance |
|---|---|---|---|---|
| METR-LA | speed | 207 | 12 | DCRNN split |
| PEMS-BAY | speed | 325 | 12 | DCRNN split |
| PEMS04 (flow / speed) | flow, speed | 307 | 12 | ASTGNN split |
| PEMS08 (flow / speed) | flow, speed | 170 | 12 | ASTGNN split |

Road adjacency is repaired to a single connected component (full component repair) so the
Laplacian has one null mode. PEMS metrics mask $y=0$ readings (missing), matching
SpecSTG/DiffSTG.

### 5.3 GAD-faithful protocol (verified against the reference code)

GAD's paper omits most hyperparameters. We reverse-engineered them from the official
implementation (github.com/vimalkb7/gad) and match every knob; this table *is* a
reproducibility contribution.

| knob | GAD reference code | GAD-ST (ours) |
|---|---|---|
| prior shift $\gamma$ | 0.8 | **0.8** |
| schedule | FCPS, $S=7,\ c_{\min}=0.1,\ \alpha=4$ | identical |
| mixing $2S\theta_{\min}$ | $\approx 11.2$ | $11.2$ |
| denoiser | GCNN, $K{=}[5,5,5]$, $C{=}24$, SiLU, residual | identical (shared by all arms) |
| shift operator | normalized sym. Laplacian | identical |
| target | $\mathbf x_0$ (Tweedie score at sampling) | identical |
| loss | SNR-weighted MSE $+\,5\!\cdot\!10^{-3}$ graph-TV | identical |
| noise sampling | $\bar c = S\cdot u^{0.65},\ u\!\sim\!\mathcal U(0,1)$ | identical |
| optimizer | AdamW, wd $10^{-4}$, grad-clip 1.0 | identical |
| MMD | RBF, 5 log-spaced bandwidths (median heuristic) | identical |
| baselines share denoiser | **yes** (graph-agnostic = the SDE, not the net) | yes |
| sampler | Euler–Maruyama reverse SDE | EM + exact ancestral |

**Key correction (validity note).** An earlier unconditional run silently used a
data-fitted *empirical* temporal covariance instead of the Cartesian prior (an argparse
default), giving a stiffness of 216 and an EM stability floor of $\sim$1208 steps; it was
then evaluated *below* that floor, so those numbers were sampler artifacts and are
discarded. The runs reported here use the Cartesian GAD-ST prior (stiffness 3, EM floor
$\sim$19) with $\gamma=0.8$, the graph-TV regularizer enabled, and evaluation in GAD's
small-step regime. This correction is documented in full for transparency.

### 5.4 Evaluation

**Unconditional (RQ1, RQ3).** We follow GAD's evaluation exactly. For each generated and
each held-out test set we compute three per-window graph-signal statistics and the maximum
mean discrepancy (MMD) between their distributions:

- **TV** (quadratic variation) $\mathbf x^\top\mathbf L_{\text{comb}}\mathbf x$ — graph
  smoothness (combinatorial Laplacian);
- **SC** (spectral centroid) — where the signal's energy sits in the graph spectrum
  (normalized Laplacian);
- **DC** (degree correlation) — Pearson correlation of the signal with node degree.

The headline number is the **average MMD (aMMD)** over these three (GAD Fig. 3). MMD uses
an RBF kernel with a 5-bandwidth mixture. We report aMMD as a function of reverse-step count
in GAD's regime — ancestral $\{5,10,25,50,100\}$ and EM $\{20,50,100,250\}$ — averaged over
3 sampling seeds with standard deviations (single MMD cells swing $\pm0.9$).

**Conditional (RQ2).** Standard probabilistic-forecasting metrics on the test split, masked
at $y=0$ for PEMS:

- **CRPS** (19-quantile normalized, CSDI/DiffSTG convention) — sharpness+calibration;
- **MAE / RMSE** — point accuracy of the posterior mean;
- **coverage / PICP and Winkler score** — interval calibration;
- the same metrics as a function of few-step ancestral counts $\{2,5,10,25\}$ to test the
  small-step claim in the forecasting regime.

### 5.5 Samplers and step counts

Training is sampler-agnostic (closed-form forward). At generation we use the exact
ancestral sampler as the primary integrator (no discretization error, so any gap reflects
the learned model) and GAD's EM sampler for parity with their figures. All reported EM step
counts exceed the $\gamma=0.8$ stability floor ($\approx$19), so no result is an
under-resolved artifact.

### 5.6 Implementation and reproducibility

Branch `raw-signal-gad`; training and evaluation on the DAIC cluster (NVIDIA A40 / RTX PRO
6000 Blackwell), PyTorch 2.7.1+cu128. Seed 42 for the main tables; multi-seed sampling for
error bars. Experiment tags: **`20260716_gadst`** (unconditional, 4 arms) and
**`20260716_gadst_cond`** (conditional residual, GAD-ST vs isotropic). Each run persists its
full config; priors are rebuilt from config at evaluation, so every number is reproducible
from the checkpoint.

---

## 5.7 Results (scaffolding — filled from the live runs)

**Table 5.1 — Unconditional generation, aMMD (↓), ancestral sampler.** GAD-ST vs VPD, VED,
and the isotropic control, per dataset and step count (mean over 3 seeds).

| dataset | steps | GAD-ST | VPD | VED | ISO | winner |
|---|---|---|---|---|---|---|
| METR-LA | 5 / 10 / 25 / 50 / 100 | … | … | … | … | … |
| PEMS08-flow | 5 / 10 / 25 / 50 / 100 | … | … | … | … | … |
| PEMS08-speed | 5 / 10 / 25 / 50 / 100 | … | … | … | … | … |

**Table 5.2 — Unconditional generation, aMMD (↓), EM sampler** ($\{20,50,100,250\}$ steps,
GAD's protocol). *(same layout).*

**Table 5.3 — Conditional forecasting.** CRPS, MAE, RMSE, coverage, Winkler for GAD-ST vs
isotropic, headline step count, per dataset.

| dataset | arm | CRPS ↓ | MAE ↓ | RMSE ↓ | cov. | Winkler ↓ |
|---|---|---|---|---|---|---|
| METR-LA | GAD-ST / ISO | … | … | … | … | … |
| PEMS08-flow | GAD-ST / ISO | … | … | … | … | … |
| PEMS08-speed | GAD-ST / ISO | … | … | … | … | … |

**Figure 5.1 — aMMD vs reverse steps** (per dataset, four curves). The small-step region is
where GAD's advantage is claimed and where any GAD-ST win over VPD/VED should appear; the
isotropic curve attributes that gap to geometry vs schedule.

---

## Interpretation guide (how each outcome reads in the thesis)

- **GAD-ST < VPD, VED, ISO at small steps** → the graph prior extends GAD to
  spatio-temporal forecasting; positive result, and the ISO control proves it is the
  geometry, not the schedule.
- **GAD-ST ≈ VPD/ISO** → under a graph-aware denoiser the noise geometry is neutral; the
  denoiser already carries the graph. A precise, controlled negative that corrects GAD's
  unmatched comparison.
- **GAD-ST loses even at GAD's settings** → structure belongs in the operator, not the
  corruption; the strongest defensible claim, with mechanism and controls the original
  paper lacks.

In all three the deliverable is the same: the **first controlled decomposition of
graph-aware diffusion's advantage** into geometry, schedule, and operator, across generation
and forecasting, on six benchmarks, with every hyperparameter matched to the reference.

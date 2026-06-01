# How the diffusion pipeline works — from residuals to samples

This is the plain-language tour of what we built and *why* each choice was made.
It follows the data the whole way through:

```
raw traffic  →  GWN backbone  →  residuals  →  Kronecker prior  →  forward SDE (FCPS)
                                                                          │
                          samples  ←  reverse sampler  ←  denoiser (UGnet)  ┘
```

Every section points at the exact file/function so you can match the words to the
code. Worked numbers use **METR-LA** as the running example: $N = 207$ sensors,
$H = 12$ forecast steps.

> **Note on the math:** equations are written in LaTeX. They render as formulas in
> GitHub, VS Code's Markdown preview, or any KaTeX/MathJax viewer. If you're reading
> raw text, `$...$` just wraps an inline formula and `$$...$$` a displayed one.

---

## 0. The one-sentence goal

We already have a deterministic forecaster (Graph-WaveNet, "GWN") that predicts a
single number per sensor per horizon. **We want a *distribution* instead of a single
number** — so we can say "traffic at sensor 42, 30 min ahead, is $55 \pm 8$ km/h" and
have that $\pm 8$ be calibrated. Diffusion models are how we learn that distribution.

But we don't diffuse the *traffic*. We diffuse the **residuals** — what GWN got
wrong. That's the first key idea.

---

## 1. Residuals — what we actually model

**Code:** produced by `scripts/extract_residuals.py`, cached in
`results/residuals/gwn_{dataset}_{split}_seed0.npz`, loaded by
`src/data/residual_dataset.py` and `src/diffusion/prior.py:_load_residuals_train`.

GWN gives a point forecast $\hat{Y}$. The residual is simply

$$
R \;=\; Y_{\text{true}} - \hat{Y}, \qquad \text{shape } (B, N, H)
$$

for $B$ windows, $N$ sensors, $H$ horizons. Why model $R$ and not $Y$ directly?

1. **GWN already nailed the easy part** (the mean / daily pattern). The leftover $R$
   is the *hard, stochastic* part — the bit that's genuinely uncertain. Spending the
   model's capacity on uncertainty rather than re-learning the mean is far more
   sample-efficient.
2. **$R$ is roughly zero-mean and roughly stationary**, which is exactly the regime
   a Gaussian prior + linear SDE is good at.

At the end we add the forecast back: $Y_{\text{sample}} = \hat{Y} + R_{\text{sample}}$.
The whole diffusion machinery lives in residual-space.

> **Example.** GWN predicts 55 km/h at one sensor 30 min out; the truth was 47. The
> residual is $-8$. Across thousands of windows these residuals have structure:
> neighbouring sensors err together, and an error now correlates with the error 5 min
> later. Capturing *that* structure is the prior's job.

A locked precondition (memory `project_section1_lockin`): residuals stay in **raw
units** (km/h), shape $(B,N,H)$, and the train-vs-val variance ratio must be within
10% before anything else proceeds.

---

## 2. The Kronecker prior — the shape of "typical" residuals

**Code:** `src/diffusion/prior.py` (`KroneckerPrior`, `load_prior`, `_fit_sigma_T`).

The prior is our claim about what a *plausible* residual field looks like before we
see data. We model it as a zero-mean Gaussian:

$$
\operatorname{vec}(R) \;\sim\; \mathcal{N}\!\left(0,\; \Sigma\right),
\qquad
\Sigma \;=\; \Sigma_T \otimes \Sigma_S .
$$

The $\otimes$ (Kronecker product) is the second key idea. Instead of one giant
$(N H)\times(N H)$ covariance — for METR-LA that's $2484\times 2484 \approx 6.2$M
entries — we **factor** it into:

- $\Sigma_S$ — the **spatial** covariance: how sensors co-vary. Size $N\times N = 207\times207$.
- $\Sigma_T$ — the **temporal** covariance: how the $H$ horizons co-vary. Size $H\times H = 12\times12$.

The bet: spatial and temporal correlation are roughly separable, so the joint
covariance is their product. Far fewer parameters, and — crucially — it
**diagonalises** (Section 3).

### 2a. Spatial factor $\Sigma_S$ — from the road graph

$$
\Sigma_S \;=\; \left(L + \gamma I\right)^{-1}
$$

$L$ is the symmetric normalised graph Laplacian of the road network (road-neighbour
sensors are coupled). $(L+\gamma I)^{-1}$ is a **Gaussian Markov random field**: high
covariance between connected sensors, decaying with graph distance. The shift
$\gamma$ controls how far correlation spreads — small $\gamma$ = long-range, large
$\gamma$ = local.

$\gamma$ is **locked per dataset** (`GAMMA_LOCKED`, `prior.py:29`), fitted once and frozen:

$$
\gamma_{\text{METR-LA}} = 0.158, \quad
\gamma_{\text{PEMS-BAY}} = 0.213, \quad
\gamma_{\text{PEMS08F}} = 0.924 .
$$

We never form $(L+\gamma I)^{-1}$ explicitly. We use the Laplacian's
eigendecomposition $L = V_S\,\operatorname{diag}(\lambda_S)\,V_S^\top$ (cached in
`data/processed/{ds}_laplacian_eigs.npz`). The eigenvalues of $\Sigma_S$ are then
$\dfrac{1}{\lambda_k + \gamma}$.

### 2b. Temporal factor $\Sigma_T$ — from the residuals themselves

$$
\Sigma_T \;=\; D_H^{1/2}\, R_H \, D_H^{1/2}
\qquad (\texttt{prior.py:\_fit\_sigma\_T})
$$

- $D_H$ = per-horizon variance, measured **robustly** with the MAD (median absolute
  deviation) so outlier windows don't blow it up:
  $$
  \delta_h \;=\; \left(1.4826 \cdot \mathrm{MAD}_h\right)^2
  \qquad (\texttt{prior.py:108})
  $$
  The factor $1.4826$ makes MAD match the standard deviation for Gaussian data.
- $R_H$ = the empirical $H\times H$ correlation matrix of residuals across horizons
  (`np.corrcoef`, `prior.py:110`).

So $\Sigma_T$ says: horizon 1 has variance X, horizon 12 has variance Y (larger —
further ahead is harder), correlated according to $R_H$.

### 2c. The joint eigen-structure $\theta$

This is the payoff. Because $\Sigma = \Sigma_T \otimes \Sigma_S$, the **precision**
(inverse covariance) $Q = \Sigma^{-1}$ shares a single eigenbasis
$U = V_S \otimes V_T$, and its eigenvalues are the outer combination
(`prior.py:135`):

$$
\theta[k, h] \;=\; \frac{\lambda_S[k] + \gamma}{\mu_T[h]},
\qquad \text{shape } (N, H),
$$

where $\lambda_S[k]$ are Laplacian eigenvalues and $\mu_T[h]$ are eigenvalues of
$\Sigma_T$. The **stationary variance of each independent mode is $1/\theta[k,h]$**
(`prior.sigma2`, `prior.py:83`).

> **Why this matters:** in the eigenbasis, the $2484$-dimensional correlated Gaussian
> becomes $2484$ *independent* 1-D Gaussians, each with variance $1/\theta$. Every
> hard step below (forward noising, reverse sampling) becomes a per-number scalar
> operation. This is the engine of the whole method.

---

## 3. The eigenbasis trick — rotate, work, rotate back

**Code:** `_rotate_to_eigenbasis` / `_rotate_from_eigenbasis`,
`forward_fcps.py:118-125` (and the torch `einsum` versions).

Any residual field $R$ (shape $N\times H$) can be written in the eigenbasis:

$$
C \;=\; V_S^\top\, R\, V_T \quad\text{(rotate to eigenbasis)},
\qquad
R \;=\; V_S\, C\, V_T^\top \quad\text{(rotate back)} .
$$

$C[k,h]$ is the coordinate of $R$ along eigenmode $(k,h)$. In $C$-space the modes are
**uncorrelated**, so noising/denoising mode $(k,h)$ only involves the single scalar
$\theta[k,h]$. The 2-D `einsum` performs both rotations at once (spatial via $V_S$,
temporal via $V_T$).

Everywhere below that you see $H_t \odot C_0$ or $\sigma^2_t \odot Z$, the $\odot$ is
**element-wise** on an $N\times H$ array — legal only because we're in the eigenbasis.

---

## 4. The forward SDE + FCPS schedule — how we add noise

**Code:** `src/diffusion/forward_fcps.py` (`FCPSSchedule`, `sample_R_t_given_R0`).

### 4a. The dynamics (from the GAD paper)

A diffusion model needs a *forward process* that gradually turns data into noise. We
use **GAD's graph-aware OU SDE**:

$$
\mathrm{d}r \;=\; -\,c(t)\,Q\,r \,\mathrm{d}t \;+\; \sqrt{2\,c(t)}\;\mathrm{d}W,
\qquad W \sim \mathcal{N}(0, I\,\mathrm{d}t).
$$

Read it as two forces on the residual $r$:

- $-\,c(t)\,Q\,r\,\mathrm{d}t$ — a pull *toward zero*, shaped by $Q$ so it pulls
  hardest along directions the prior thinks should be small.
- $\sqrt{2\,c(t)}\;\mathrm{d}W$ — random kicks (Brownian noise).

Run it long enough and $r$ forgets its start and settles into the stationary
distribution — which is **exactly our prior** $\mathcal{N}(0, \Sigma_T\otimes\Sigma_S)$
by construction. That is the design: forward-process noise *is* the structured prior,
not white noise. (That's what makes it "graph-aware" vs. plain isotropic DDPM.)

$c(t)$ is a *speed knob*: how fast we run the clock at time $t$.

### 4b. FCPS — the schedule for $c(t)$

**Floor-Constrained Polynomial Scheduler** (`FCPSSchedule`, `forward_fcps.py:45`).
With $u = t/T$:

$$
c(t) \;=\; c_{\min} + k\,u^{\alpha},
\qquad
k \;=\; \frac{(c_0 - c_{\min} T)(\alpha + 1)}{T}.
$$

Its time-integral $\bar{c}(t) = \int_0^t c(s)\,\mathrm{d}s$ — the quantity that
actually drives the noising — is

$$
\boxed{\;\bar{c}(t) \;=\; c_{\min}\,t \;+\; (c_0 - c_{\min} T)\,u^{\alpha + 1}\;}
\qquad\Longrightarrow\qquad
\bar{c}(T) = c_0 .
$$

Defaults: $c_{\min}=0.05$, $c_0=20000$, $\alpha=7$, $T=1$.

In words: keep the clock **slow** for most of $[0,T]$ (the $c_{\min}$ floor), then
**ramp up steeply near $t=T$** (the $u^7$ term). Why? The slow eigenmodes (small
$\theta$, large variance — long-range spatial structure) mix very reluctantly. The
steep late ramp dumps a huge amount of "diffusion time" $\bar c$ right at the end to
force even those modes to fully randomise. ($\alpha > 1$ is required —
`__post_init__`, line 57.)

> **Worked numbers (METR-LA defaults).**
> $\bar c(0) = 0$,
> $\;\bar c(0.5) = 0.05\cdot 0.5 + 19999.95 \cdot 0.5^{8} \approx 0.025 + 78 \approx 78$,
> $\;\bar c(1.0) = 20000$.
> So half the *clock* time has passed by $t=0.5$, yet only $\sim 0.4\%$ of the
> *diffusion* $\bar c$ has — almost all noising happens in the last sliver of $t$.

> ⚠️ **Open question — is $\alpha=7$ too aggressive?** Training samples
> $t\sim\mathrm{Uniform}$, so the *induced* distribution over $\bar c$ inherits this
> extreme skew. Verified numerically (`scripts/section3/verify_sampler.py`, test D):
> with $\alpha=7$, **56%** of training draws land at $\bar c < 1\%$ of $c_0$ (median
> $\bar c/c_0 = 0.004$) — the network sees mostly *near-clean* residuals and rarely
> the mid-noise regime. Dropping to $\alpha=1.5$ (GAD's value) cuts that to 16%
> (median $0.18$). This is a real train-time concern: the denoiser may be
> under-trained at the noise levels the reverse sampler spends most of its *useful*
> work in. Candidate fixes for Stage D: lower $\alpha$, or importance-sample $t$ so
> $\bar c$ is closer to uniform. Not yet applied.

### 4c. The closed-form marginal (no simulation needed)

Because each eigenmode is a scalar linear (Ornstein–Uhlenbeck) SDE, we can jump
straight to $R_t$ at any $t$ **without stepping through time**
(`sample_R_t_given_R0`, `forward_fcps.py:153-162`). Per mode $(k,h)$, in the
eigenbasis:

$$
\begin{aligned}
H_t        &= \exp\!\big(-\bar{c}(t)\,\theta\big)              &&\text{mean-decay factor (line 155)}\\[2pt]
\sigma^2_t &= \frac{1 - \exp\!\big(-2\,\bar{c}(t)\,\theta\big)}{\theta} &&\text{added variance (line 157)}\\[4pt]
C_t        &= H_t \odot C_0 \;+\; \sqrt{\sigma^2_t}\odot Z,\quad Z\sim\mathcal N(0,I) &&\text{(line 161)}
\end{aligned}
$$

- At $t=0$: $\bar c = 0 \Rightarrow H_t = 1,\ \sigma^2_t = 0 \Rightarrow C_t = C_0$
  (no noise, exact). ✅
- At $t=T$ with large $\bar c$: $H_t \to 0,\ \sigma^2_t \to 1/\theta \Rightarrow C_t$
  is a pure prior draw. ✅

This is what makes training cheap: pick a random $t$, get $R_t$ in one shot. The torch
version (`sample_R_t_given_R0_torch`, line 216) does the same with per-batch-element
$t$, on GPU, every step.

### 4d. The mixing diagnostic

**Code:** `scripts/section3/mixing_diagnostic.py` → `results/diffusion/mixing_diagnostic.json`.

If the slowest mode ($\theta_{\min}$) hasn't mixed by $t=T$, the forward process never
reaches the prior and the model is biased. The check:

$$
\theta_{\min}\cdot \bar c(T) \;\ge\; 5
\qquad\big(\text{mean attenuation } e^{-5} < 1\%\big).
$$

This is *why* $c_0$ defaults to the large $20000$: it was bumped until all three
datasets pass (PEMS08F is tightest at $\approx 11.3$). `FCPSSchedule.for_prior`
(line 77) can auto-pick $c_0$ from a prior's $\theta_{\min}$.

> **The threshold $5$ is a *choice*, not a law.** $\theta_{\min}\bar c(T)\ge 5$ means
> the slowest mode keeps $e^{-5}=0.7\%$ of its initial mean at $t=T$. We could relax
> to $\ge 1$ ($e^{-1}=37\%$ residual) and accept a small bias in that one mode —
> which would cut $c_0$ by $5\times$. Worth a Stage-D ablation (does looser mixing hurt
> CRPS/PIT?), especially since the reverse sampler conditions strongly on $\hat Y$ and
> may absorb mild under-mixing. **But note:** relaxing the threshold only rescales
> $c_0$ — it does *not* change the stiffness story in §8, because the EM stability
> number $c\,\theta\,\delta$ is invariant to that rescaling (see §5).

That large $c_0$ deserves its own explanation — next section.

---

## 5. Why is $c_0$ so huge? (And is something wrong?)

Short answer: **it's a units artifact, not a bug.** The math is exact regardless of
the number. But it *is* the visible symptom of a real property — **stiffness** — which
is the actual reason naïve Euler–Maruyama fails in Section 8.

**Where 20000 comes from.** From the mixing condition, $c_0 = \bar c(T) \ge 5/\theta_{\min}$, and

$$
\theta_{\min} \;=\; \frac{\lambda_{\min} + \gamma}{\mu_{\max}}
\;\approx\; \frac{\gamma}{\mu_{\max}}
\qquad(\lambda_{\min}\approx 0).
$$

- $\gamma_{\text{METR-LA}} = 0.158$ — small.
- $\mu_{\max}$ = the largest eigenvalue of $\Sigma_T$. But $\Sigma_T$ is built from
  residual variances in **raw km/h²**. Residuals are $\sim 5\text{–}8$ km/h, so
  variances are tens, and the top eigenvalue of the $12\times12$ $\Sigma_T$ aggregates
  correlated variance across horizons → **a few hundred**.

So $\theta_{\min} \approx 0.158 / (\text{few hundred}) \approx 5\times10^{-4}$, hence

$$
c_0 \;\ge\; \frac{5}{\theta_{\min}} \;\approx\; \frac{5}{5\times10^{-4}} \;=\; 10^4 .
$$

**The key idea — it's about units.** $\theta$ has units of $1/\text{variance}$, so
$\bar c$ must have units of *variance* for the product $\bar c\,\theta$ (the only thing
that physically matters) to be dimensionless. $c_0 = 20000$ is "20000 km/h²" — it
looks enormous only because variance in raw km/h² is numerically large. The
dimensionless products $\bar c\,\theta$ are perfectly sensible ($\ge 5$ for slow
modes).

**Does it cause problems? Stage by stage:**

| Stage | Hurt by large $c_0$? | Why |
|---|---|---|
| Forward sampling (§4) | **No** | Closed form uses $\exp(-\bar c\,\theta)$; exponent always negative → no overflow. Underflow to 0 just means "fully mixed" — correct. |
| Training (§6/§7) | **No** | Same closed form, on GPU. |
| Reverse **Euler–Maruyama** (§8) | **Yes — this is what broke** | EM stability needs $c\,\theta\,\delta \lesssim 2$; the *fast* modes have huge $\theta$, so $c\,\theta_{\max}\,\delta \approx 500$. Blows up. |
| Reverse **exact / DDIM** (§8) | **No** | Exact transition, no step-size limit. |

**The real thing you sensed.** It's not a bug — but you're right that *something* is
extreme. It's the **dynamic range of $\theta$**: $\theta_{\max}/\theta_{\min}$ spans
many orders of magnitude (a *stiff* spectrum). That's intrinsic to a **structured**
prior — some modes are tightly pinned, some nearly free — and it's the true cause of
the EM blow-up. The big $c_0$ is just the messenger.

**Could we make $c_0$ look normal? And would that fix the EM blow-up in §8?**
We *can* make it look normal — normalise $\Sigma_T$ to unit trace (and standardise the
residuals to match) and $c_0$ drops to $\mathcal O(10\text{–}100)$. **But it will not
fix EM**, and here is the proof. Rescaling $\Sigma_T \to \Sigma_T / s$ does three
things in lockstep:

$$
\theta \;\to\; s\,\theta, \qquad
c_0 = \tfrac{5}{\theta_{\min}} \;\to\; \tfrac{c_0}{s}, \qquad
c(t) \;\to\; \tfrac{c(t)}{s}.
$$

Now look at the quantities that actually govern behaviour:

| Quantity | Scales as | Net |
|---|---|---|
| forward exponent $\bar c\,\theta$ | $(c/s)\cdot(s\theta)$ | **invariant** |
| stiffness ratio $\theta_{\max}/\theta_{\min}$ | $(s\theta_{\max})/(s\theta_{\min})$ | **invariant** |
| EM stability number $c\,\theta\,\delta$ | $(c/s)\cdot(s\theta)\cdot\delta$ | **invariant** |

Every dimensionless quantity — including the exact arguments fed to $\exp(\cdot)$ — is
**scale-invariant**. So normalisation moves the *magnitude* of $c_0$ around but leaves
the dynamics, the float-precision arguments, and EM's stability **identical**. This is
confirmed numerically (`verify_sampler.py`, test B): a stiff mode blows EM up at any
practical step count regardless of how $c_0$ and $\theta$ are split.

**Verdict:** normalising is worth doing for **readability** (a defense audience won't
trip over "20000") and to guard the $\mu_h \ge 10^{-12}$ clip in `prior.py:133` from
producing absurd $\theta_{\max}$ — but it is **not** a numerical fix and it does **not**
rescue Euler–Maruyama. The reason we sample stably is the choice of update in §8, not
the scale of $c_0$.

---

## 6. The denoiser — the only learned part

**Code:** `src/diffusion/denoiser.py` (`R0Denoiser`), wrapping DiffSTG's `UGnet`.

Everything above is fixed math. The *only* thing we train is a network that, given a
noisy residual $R_t$ and the time $t$ (plus the forecast $\hat Y$), predicts the clean
residual:

$$
\hat R_0 \;=\; \text{denoiser}\big(R_t,\; t,\; \hat Y\big),
\qquad \text{all shape } (B, N, H)
\qquad (\texttt{denoiser.py:129}).
$$

### What we change in DiffSTG's UGnet — and why

**The honest answer: we change *nothing inside* UGnet's source.** We import it
unchanged (`denoiser.py:47`) and wrap it in `R0Denoiser`. Every "change" lives in our
wrapper — how we feed the network and what we train it against. There are exactly four,
three plumbing and one conceptual:

| # | Change | Where | Why |
|---|---|---|---|
| 1 | **Input plumbing**: expose a clean `forward(R_t, t, Ŷ)`; internally map to UGnet's `x:(B,F,V,T)` + `c=(x_masked, pos_w, pos_d)`. Unsqueeze a feature channel, feed $\hat Y$ into the `x_masked` slot, pass dummy zeros for the unused `pos_w/pos_d`. | `denoiser.py:156-160` | Shape compatibility, and give the network the backbone forecast $\hat Y$ as conditioning. |
| 2 | **Scale time by $\times 1000$** (`T_EMBED_SCALE`) | `denoiser.py:54,162` | UGnet's sinusoidal time embedding was built for *integer* DDPM steps in $[0,\sim\!1000]$. Our $t$ is continuous in $[0,1]$. Multiplying by 1000 puts it in the frequency range the embedding expects. |
| 3 | **Config $T_p = H,\ T_h = 0$** | `denoiser.py:74-75` | DiffSTG splits time into $T_h$ history + $T_p$ prediction steps (internal dim $T_p + T_h$). We feed only the $H$-horizon residual — no history channel — so all $H$ go in $T_p$. The internal concat with `x_masked` then gives $2H$, and UGnet's final $\text{Linear}(2H, H)$ restores the horizon. (Found while tracing the `16x24 vs 48x24` shape bug.) |
| 4 | **Target $= R_0$, not $\varepsilon$** (Tweedie) | *not in the network* — in the loss, `train_diffusion.py` | DiffSTG trains UGnet to predict the *noise* $\varepsilon$. We train the *same output* to predict the *clean residual* $\hat R_0$. Architecture is byte-for-byte identical; only the regression target differs. We chose $R_0$-prediction because (a) our noise covariance varies wildly across modes (stiffness again) and $R_0$-prediction is better-conditioned there, and (b) the reverse sampler is written in terms of $\hat R_0$. |

So **#1–#3 are adapter glue and #4 is a loss-function choice — UGnet's `.py` is
untouched.** This works because UGnet is "just" a map $(\text{noisy grid}, t,
\text{conditioning}) \to \text{grid}$; whether you call the output $\varepsilon$ or
$\hat R_0$ is decided entirely by your training target.

UGnet itself is a U-Net with temporal convolutions + graph convolution + a sinusoidal
time embedding — purpose-built for spatiotemporal grids. About 1M parameters per
dataset.

The two adapters to keep an eye on (also flagged in the file header):
- **#2 ($\times1000$):** if real training stagnates, the time embedding's resolution is
  the first suspect.
- **#3 ($T_h=0$):** benign, but it's why we currently drop the "X history" conditioning
  DiffSTG uses — the deferred "wire up X history" item.

---

## 7. Training — fit the denoiser

**Code:** `scripts/train_diffusion.py`; smoke version `scripts/section3/train_synthetic_smoke.py`.

One training step is just the forward marginal + a regression loss:

$$
\begin{aligned}
&1.\ \text{sample a batch of real } R_0,\ \text{with their } \hat Y \text{ and mask}\\
&2.\ t \sim \mathrm{Uniform}[t_\varepsilon,\, T] \quad\text{(one } t \text{ per window)}\\
&3.\ R_t = \texttt{sample\_R\_t\_given\_R0\_torch}(R_0, t, \text{prior}, \text{schedule}) \quad(\S4c)\\
&4.\ \hat R_0 = \text{denoiser}(R_t, t, \hat Y) \quad(\S6)\\
&5.\ \mathcal L = \mathrm{masked\_MSE}(\hat R_0,\, R_0) \quad\text{(ignore missing sensors)}\\
&6.\ \text{Adam step, grad-clip } 5.0
\end{aligned}
$$

No SDE simulation in the loop — the closed form does it all. Checkpoints land in
`results/checkpoints/diffusion/{run_id}/`; validation averages 4 $t$-draws to cut
variance.

The **synthetic smoke test** draws $R_0$ *from the prior itself* (so data exactly
matches model assumptions) and checks the loss drops below the zero-predictor
baseline — a cheap wiring check before spending GPU hours.

---

## 8. Reverse sampling — and why DDIM over Euler–Maruyama

**Code:** `src/diffusion/reverse.py` (`reverse_sample`).

To *generate* a residual we start from pure prior noise at $t=T$ and walk back to
$t=0$, calling the denoiser at each step.

### 8a. What the GAD paper writes: the reverse SDE

The textbook reverse process is another SDE, solved with **Euler–Maruyama (EM)** — the
simplest integrator: take a small step $\delta$, move by $\text{drift}\cdot\delta$, add
$\text{noise}\cdot\sqrt{\delta}$, repeat. Per eigenmode:

$$
\mathrm{d}c_k \;=\; \big[-c(t)\,\theta_k\, c_k \;-\; 2\,c(t)\,\nabla\!\log p_t\big]\,\mathrm{d}t
\;+\; \sqrt{2\,c(t)}\;\mathrm{d}\bar W .
$$

### 8b. Why naïve EM **explodes** for our settings

EM is stable only if each step's effective decay is small:

$$
c(t)\,\theta\,\delta \;\lesssim\; 2 \qquad\text{(EM stability boundary).}
$$

But recall §4d/§5: we were forced to set $c_0 = 20000$ to mix the slow modes. That same
large $c$ makes the **fast** modes brutally stiff. For the largest $\theta$,

$$
c\,\theta_{\max}\,\delta \;\approx\; 500 \;\approx\; 250\times \text{ the stability limit.}
$$

EM overshoots, oscillates, and diverges to `NaN` within a few steps. We saw exactly
this.

> **Analogy.** EM here is like simulating a stiff spring with too large a timestep:
> instead of settling, each step overshoots further than the last until it flies off to
> infinity. You *can* fix it with tiny steps — but you'd need
> $\sim \theta_{\max}\bar c(T)/2$, i.e. tens of thousands of steps. Wasteful.

### 8c. The Tweedie link — from $\hat R_0$ to the score

First, why predicting $\hat R_0$ even lets us run the reverse SDE, which is written in
terms of the *score* $\nabla\log p_t$. **Tweedie's formula** ties them together. For
our eigenbasis marginal $C_t = H_t c_0 + \mathcal N(0,\sigma^2_t)$,

$$
\mathbb E[c_0 \mid C_t] \;=\; \frac{C_t + \sigma^2_t\,\nabla\log p_t(C_t)}{H_t}
\qquad\Longleftrightarrow\qquad
\nabla\log p_t(C_t) \;=\; \frac{H_t\,\hat C_0 - C_t}{\sigma^2_t}.
$$

So the denoiser's $\hat C_0 = \mathbb E[c_0\mid C_t]$ *is* the score, rescaled. The
$\hat\varepsilon$ we back out below is just $-\sigma_t\,\nabla\log p_t$. The practical
chain is therefore

$$
\hat R_0 \;\xrightarrow{\text{Tweedie}}\; \text{score} \;\xrightarrow{\text{plug into}}\; \text{reverse update},
$$

i.e. predicting the clean residual is a numerically friendlier parameterisation of
predicting the score — not a different object.

### 8d. The fix: a *stable* discretization (this is the DDIM/ancestral update)

The crucial structural fact: **each eigenmode is a *linear* SDE, whose conditional
bridge $p(C_{t-\delta}\mid C_t, c_0)$ is Gaussian in closed form.** Rather than
Euler-step the SDE, we sample that bridge directly. In the eigenbasis
(`reverse.py:158-189`), each reverse step is:

$$
\begin{aligned}
&\text{1. denoiser predicts } \hat R_0 \text{ at the current } t\\
&\text{2. rotate } R_t,\hat R_0 \text{ to eigenbasis} \to C_t,\ \hat C_0\\
&\text{3. back out the noise:}\quad
   \hat\varepsilon = \frac{C_t - H_{\text{cur}}\odot \hat C_0}{\sqrt{\sigma^2_{\text{cur}}}}
   &&(\texttt{line 172})\\
&\text{4. step to } t-\delta:\quad
   C_{\text{next}} = H_{\text{next}}\odot\hat C_0
   + \sqrt{\text{drift\_var}}\odot\hat\varepsilon
   + \sqrt{\tilde\sigma^2}\odot\zeta
   &&(\texttt{line 186-189})
\end{aligned}
$$

where $H$ and $\sigma^2$ are the *same* closed-form factors as the forward marginal
(`_eigenbasis_HSigma`, line 73). The $\eta\in[0,1]$ knob (`reverse.py:181`) sets how
much fresh noise each step injects: $\eta=0$ is the deterministic probability-flow ODE
(repeatable trajectories), $\eta=1$ is the full stochastic ancestral level.

I verified algebraically that this update's mean **and** variance equal the *exact* OU
bridge $p(C_{t-\delta}\mid C_t, c_0)$. So far so good — but read the next part
carefully, because it corrects an earlier overclaim.

### 8e. Honest framing — *stable*, not *exact* (verified numerically)

It is tempting to call this "exact integration, no discretization error." **That is
wrong, and `scripts/section3/verify_sampler.py` proves it.** The bridge formula is
exact *only if you know the true $c_0$* — but in sampling we don't, so we plug in the
posterior **mean** $\hat C_0 = \mathbb E[c_0\mid C_t]$. Substituting the mean for the
full posterior discards its spread, which biases the sampled variance **low** at finite
step counts. This is the *universal* diffusion-sampling approximation (DDPM, DDIM, all
of them share it), not something specific to us.

What the verification shows:

| Property | EM (GAD's literal solver) | Our $\eta$-ancestral update |
|---|---|---|
| Blows up when $c\theta\delta \gg 2$? | **Yes** — NaN at any practical $n$ (test B: $c\theta=8\!\times\!10^6$ needs $n\approx 4$M) | **No** — always finite (test B) |
| Exact at finite steps? | No | **No** (test A: Var $0.66\to1.41\to1.86\to1.99$ as $n=5\to1000$) |
| Bias source | discretization **+** plug-in | plug-in (posterior mean for $c_0$) |
| Converges to truth as $n\to\infty$? | Yes | Yes — both reach the *same* limit (test C: 6% gap at $n=100$ vs fine EM) |

So the correct statement is: **$\eta$-ancestral and EM are two discretizations of the
*same* GAD reverse SDE.** They converge to the same answer as $n\to\infty$; the
difference is that our update is **unconditionally stable** while EM is not. We choose
it for stability, *not* exactness.

**Practical consequence for the thesis (Stage D).** The finite-step bias is a slight
**under-dispersion** → mild PIT under-coverage that *shrinks with more steps*. So
`n_steps` is effectively a **calibration knob**: sweep it (e.g. 100 / 250 / 500) and
read PIT. One tempting "fix" — spacing the reverse grid uniformly in $\bar c$ instead
of in $t$ — was tested (test E) and is **worse** ($-22\%$ vs $-6\%$ bias), so the grid
stays uniform-in-$t$; bump `n_steps` instead. Honest phrasing for the writeup:
*"stable ancestral sampling of the GAD reverse SDE; we verify convergence in step count
and report the operating point,"* with EM kept as an ablation baseline that
demonstrably diverges under the stiff schedule.

At the final step $H_0 = 1,\ \sigma^2_0 = 0$, so the returned value is the denoiser's
last $\hat R_0$ (`reverse.py:118-121`).

---

## 9. End to end, in one breath

$$
\begin{aligned}
R &= Y_{\text{true}} - \hat Y &&\text{residuals (§1)}\\
\Sigma &= \Sigma_T \otimes \Sigma_S,\quad \theta[k,h] = \tfrac{\lambda_k + \gamma}{\mu_h} &&\text{Kronecker prior (§2)}\\
C &= V_S^\top R\, V_T &&\text{rotate to eigenbasis (§3)}\\
\text{forward: } C_t &= e^{-\bar c\theta}\odot C_0 + \sqrt{\tfrac{1 - e^{-2\bar c\theta}}{\theta}}\odot Z &&\text{FCPS (§4)}\\
\text{train: } \hat R_0 &= \text{UGnet}(R_t, t, \hat Y),\ \ \mathcal L = \text{MSE} &&\text{denoiser (§6–7)}\\
\text{reverse: } C_{t-\delta} &= H_{\text{next}}\odot\hat C_0 + \cdots\ \text{(stable ancestral)} &&\text{DDIM/ancestral (§8)}\\
Y_{\text{sample}} &= \hat Y + V_S\, C_0\, V_T^\top &&\text{add forecast back}
\end{aligned}
$$

## Glossary / quick map

| Symbol | Meaning | Code |
|---|---|---|
| $R,\ R_0$ | residual field $(B,N,H)$ = truth $-$ GWN forecast | `residual_dataset.py` |
| $\hat Y$ | GWN point forecast (conditioning) | `denoiser.forward` |
| $\Sigma_S = (L+\gamma I)^{-1}$ | spatial covariance from road graph | `prior.py` |
| $\Sigma_T = D^{1/2} R_H D^{1/2}$ | temporal covariance from residuals (MAD-based) | `_fit_sigma_T` |
| $\theta[k,h] = (\lambda_k+\gamma)/\mu_h$ | joint precision eigenvalue; variance $= 1/\theta$ | `prior.py:135` |
| $V_S,\ V_T$ | spatial / temporal eigenvectors (rotation) | `prior.py` |
| $c(t),\ \bar c(t)$ | FCPS speed + its time-integral; $\bar c(T)=c_0$ | `FCPSSchedule` |
| $H_t = e^{-\bar c\theta},\ \sigma^2_t$ | forward marginal mean-decay + variance | `sample_R_t_given_R0` |
| $\hat R_0$ | denoiser's clean-residual prediction (Tweedie) | `R0Denoiser` |
| $\eta$ | reverse stochasticity: 0 = ODE, 1 = GAD ancestral SDE | `reverse_sample` |

**The big $c_0$ in one line:** $\theta$ is in units of $1/\text{variance}$ and our
residuals live in raw km/h², so $\theta_{\min}$ is tiny and $c_0 = 5/\theta_{\min}$
must be large — a units artifact, harmless to the exact math.

**The DDIM decision in one line:** that same large $c_0$ makes the reverse SDE far too
stiff for Euler–Maruyama ($c\,\theta\,\delta \approx 500 \gg 2$), so we sample each
linear eigenmode's Gaussian bridge with the DDIM/$\eta$-ancestral update — the *same*
SDE, **unconditionally stable** (it never blows up) though, like every diffusion
sampler, only exact as $n_{\text{steps}}\to\infty$ (verified in
`scripts/section3/verify_sampler.py`).

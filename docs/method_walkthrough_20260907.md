# From the road graph to the numbers — a step-by-step walkthrough of what we are running

*Written 2026-09-07 against the code as it runs in the `20260907_matched` campaign
(`src/diffusion/{prior,forward_fcps,reverse,losses,denoiser}.py`, `scripts/{train,eval}_diffusion.py`).
Every formula below is the one in the code; every number quoted is either a fixed setting or
was measured on the fresh residuals built today.*

---

## 0. The story in one paragraph

A good forecaster (Graph WaveNet) predicts the next hour of traffic at every sensor. It is
wrong by some amount at every sensor and every future step — that is the **residual**. We
want a *distribution* over those errors, not one number, so we can say "the next 30 minutes
on this corridor will be between 55 and 70 km/h with 90 % probability", and — more
importantly — "the whole network's total flow will be between A and B". A diffusion model
learns that distribution by first *destroying* real residuals with noise and then learning
to *undo* the destruction. The single design choice this thesis is about is **what noise**.
The standard choice is white noise: every sensor and every horizon gets independent noise.
Our choice is noise that already has the *shape* of traffic errors: correlated along the
horizon (an error at minute 15 persists to minute 20), correlated along the road graph
(neighbouring sensors err together), and with a **shared network-wide component** (on a bad
day the whole city runs slow). Mathematically this is a Gaussian with a Kronecker covariance
$\Sigma_T \otimes \Sigma_S$ whose spatial half is the heat kernel of the road graph plus one
weak all-to-all edge. Per sensor and per step the two choices forecast about equally well —
the backbone already fixed that. On anything *joint* (network total, cumulative flow at a
sensor) white noise is wrong by construction, because it forgets that errors move together,
and the structured noise is right.

---

## 1. The data and the graph

For each dataset we have a signal $y_{i,t}$ (speed in km/h or flow in vehicles/5 min) at
$N$ sensors for a few months at 5-minute resolution, and a road graph.

| dataset | quantity | $N$ sensors | windows train / val / test | components before → after repair |
|---|---|---|---|---|
| metr-la | speed | 207 | 23 974 / 3 425 / 6 850 | 2 → 1 |
| pems-bay | speed | 325 | 36 465 / 5 209 / 10 419 | 7 → 1 |
| pems04 (f = flow, s = speed) | flow, speed | 307 | 10 173 / 3 375 / 3 375 | 12 → 1 |
| pems08 (f, s) | flow, speed | 170 | 10 691 / 3 548 / 3 548 | 1 → 1 |

A **window** is one forecasting problem: the past $P = 12$ steps (one hour) as input, the
next $H = 12$ steps (one hour) as target,

$$
X \in \mathbb{R}^{N \times P} \ \text{(known past)}, \qquad Y \in \mathbb{R}^{N \times H} \ \text{(future to predict)} .
$$

Missing values (sensor offline, recorded as $0$) are tracked in a **mask** $M \in \{0,1\}^{N\times H}$;
a masked entry is never scored and never used to fit anything.

### 1.1 The road graph

The graph is a weighted adjacency $A \in \mathbb{R}^{N\times N}$: for METR-LA and PEMS-BAY it is
the DCRNN Gaussian kernel of road distances (thresholded), for PEMS04/08 it is the
ASTGNN 0/1 adjacency of the distance file. Two clean-ups are applied before anything else:

1. **Self-loops removed**, $A_{ii} = 0$.
2. **Component repair.** PEMS04 arrives in 12 disconnected pieces, PEMS-BAY in 7, METR-LA in 2.
   A heat process on a disconnected graph never mixes across the pieces (there would be
   several "constant" directions with zero Laplacian eigenvalue, and the prior would be
   degenerate there). So each small component is joined to the giant one by **one edge**,
   between the pair of sensors whose training signals are most correlated, with weight equal
   to the median existing edge weight. After repair every graph is one connected component
   (`scripts/check_connectivity.py`: components = 1, zero-eigenvalue multiplicity = 1,
   Fiedler value > 0 on all six).

### 1.2 The graph Laplacian — the object the prior is built from

With degrees $d_i = \sum_j A_{ij}$ and $D = \mathrm{diag}(d)$, the **normalised Laplacian** is

$$
L \;=\; I \;-\; D^{-1/2} A\, D^{-1/2}, \qquad L = V \,\mathrm{diag}(\lambda_1,\dots,\lambda_N)\, V^{\!\top},\quad 0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N \le 2 .
$$

*In words:* $L$ is symmetric, so it has an orthonormal set of eigenvectors $V$ ("graph
frequencies"). Eigenvector 1 ($\lambda_1 = 0$) is the smooth *constant* pattern
(every sensor the same); eigenvectors with small $\lambda$ are smooth patterns (neighbours
alike); eigenvectors with large $\lambda$ are rough patterns (neighbours opposite). For a
signal $x$ on the graph,

$$
x^{\!\top} L\, x \;=\; \tfrac12 \sum_{i,j} A_{ij}\Big(\tfrac{x_i}{\sqrt{d_i}} - \tfrac{x_j}{\sqrt{d_j}}\Big)^2
$$

measures *how rough* $x$ is across the edges. This is the whole reason a Laplacian is the
right building block: **"smooth across the road network" becomes "small $x^{\!\top}Lx$", and a
Gaussian whose precision is built from $L$ prefers smooth patterns.**

---

## 2. Step 1 — the backbone and the residual

A Graph WaveNet (≈309 k parameters, 100 epochs, best epoch by validation MAE, one per
dataset, seed 42) produces the point forecast

$$
\hat Y = \mathrm{GWN}(X) \in \mathbb{R}^{N\times H}, \qquad R \;=\; Y - \hat Y \quad\text{(the residual, } N\times H\text{)} .
$$

Everything downstream works on $R$, never on the raw signal. Why: the raw signal has a huge
deterministic part (the daily profile) that the backbone already captures; what is left is the
*uncertainty*, and that is what we want to model as a distribution. The fresh backbones match
the earlier ones (test MAE: metr-la 2.96, pems04f 18.74, pems04s 1.60, pems08f ≈ 15, pems08s
≈ 1.3; residual standard deviations within 2 % of the archived run).

The residual caches (`residuals/<dataset>/gwn/seed42/{train,val,test}.npz`) hold $R$, $\hat Y$,
the mask, and the scalers. The diffusion stage reads only these; **iso and kron see exactly
the same residuals**.

---

## 3. Step 2 — normalisation (per sensor)

Sensors differ 3–4× in how wrong the backbone is (a ramp sensor vs. a free-flowing one). We
therefore work in **unit space**: each sensor is divided by its own training-residual
standard deviation,

$$
\sigma_i \;=\; \mathrm{std}\big(R_{i,\cdot}\ \text{over the training windows, observed entries}\big), \qquad
r_{i,h} \;=\; \frac{R_{i,h}}{\sigma_i}.
$$

(`norm=pernode`; $\sigma_i$ is clipped at 10 % of the global std so a nearly-always-masked
sensor cannot explode.) The diffusion runs on $r$ (every sensor now has variance ≈ 1 on
train) and the last step multiplies back by $\sigma_i$. The conditioning inputs — the
forecast $\hat Y$ and the observed history $X$ — are z-scored with the *signal* mean/std
(the scaler the backbone used), giving $\hat Y_z$ and $X_z$.

---

## 4. Step 3 — the noise prior (the heart of the thesis)

A diffusion model needs a **prior**: the Gaussian $\mathcal N(0, \Sigma)$ that the forward
process drives every residual *towards*, and that the reverse process *starts from*. The
denoiser then learns how to walk from a draw of $\mathcal N(0,\Sigma)$ back to a plausible
residual. The two arms differ only in $\Sigma$.

### 4.1 Arm A — isotropic (the control)

$$
\Sigma_{\text{iso}} \;=\; \sigma^2 I_{NH}, \qquad \sigma^2 = \text{mean of } r_{i,h}^2 \text{ over observed training entries.}
$$

Every sensor and every horizon gets its own independent noise of the same size. Same total
noise energy as the structured arm, zero structure.

### 4.2 Arm B — Kronecker: $\Sigma = \Sigma_T \otimes \Sigma_S$

Write the residual matrix $r \in \mathbb{R}^{N\times H}$ as one long vector
$\mathrm{vec}(r)$ of length $NH$ (horizon-major: entry $hN + i$ is $r_{i,h}$). A Kronecker
covariance says: *the covariance between $(i,h)$ and $(j,h')$ is a spatial part times a
temporal part,*

$$
\mathrm{Cov}\big(r_{i,h},\, r_{j,h'}\big) \;=\; (\Sigma_T)_{h h'} \cdot (\Sigma_S)_{ij} .
$$

**A tiny concrete case.** Two sensors ($N=2$, neighbours, spatial correlation 0.5) and two
horizons ($H=2$, temporal correlation 0.9, second horizon twice the variance):

$$
\Sigma_S = \begin{pmatrix} 1 & 0.5\\ 0.5 & 1\end{pmatrix},\quad
\Sigma_T = \begin{pmatrix} 1 & 0.9\sqrt2\\ 0.9\sqrt2 & 2\end{pmatrix},\quad
\Sigma_T\otimes\Sigma_S =
\begin{pmatrix}
1\cdot\Sigma_S & 0.9\sqrt2\cdot\Sigma_S\\
0.9\sqrt2\cdot\Sigma_S & 2\cdot\Sigma_S
\end{pmatrix}
=\begin{pmatrix}
1 & .5 & 1.27 & .64\\
.5 & 1 & .64 & 1.27\\
1.27 & .64 & 2 & 1\\
.64 & 1.27 & 1 & 2
\end{pmatrix}.
$$

Read it: sensor 1 at horizon 1 (row 1) is correlated 0.5 with sensor 2 at the same horizon,
strongly with itself one step later (1.27), and 0.64 with its neighbour one step later. Each
block is "the spatial pattern", scaled by "how the horizons relate". With $N = 307$, $H = 12$
this is a $3684\times3684$ matrix we never form: we only ever need its two factors and their
eigen-decompositions.

#### 4.2.1 The temporal factor $\Sigma_T$ — fitted from the residuals

$$
\delta_h = \text{mean of } r_{i,h}^2 \ \text{(per horizon, over observed entries)}, \qquad
(R_H)_{hh'} = \text{Pearson correlation of } r_{\cdot,h} \text{ and } r_{\cdot,h'} \ \text{(fully observed rows)},
$$
$$
\Sigma_T \;=\; D_H^{1/2}\, R_H\, D_H^{1/2}, \qquad D_H = \mathrm{diag}(\delta_1,\dots,\delta_H).
$$

So $\Sigma_T$ has **the per-horizon noise variance on its diagonal** (errors grow 1.7–2.9×
from 5 min to 60 min) **and the Pearson correlation between horizons off the diagonal**
(adjacent horizons ≈ 0.9). We eigen-decompose $\Sigma_T = V_T\,\mathrm{diag}(\mu_1,\dots,\mu_H)\,V_T^{\!\top}$
and apply a small ridge toward its own mean, $\mu_h \leftarrow (1-\rho)\mu_h + \rho\,\bar\mu$
with $\rho = 0.1$ (`SHRINK_T`), which keeps the trace and tames the stiffest direction.

#### 4.2.2 The spatial factor $\Sigma_S$ — the road graph's heat kernel plus one global edge

Step (a): **the global edge.** Before building the Laplacian we add a weak edge between
*every* pair of sensors,

$$
A_\beta \;=\; A_{\text{road}} \;+\; \beta\,\bar w\,\big(\mathbf 1\mathbf 1^{\!\top} - I\big), \qquad \bar w = \text{mean road-edge weight},
$$

with $\beta$ of order $0.003$–$0.03$ — i.e. each all-to-all link is 0.3–3 % of a typical road
link. Why this is needed (measured, `docs/…global_edge…`): the residual has a **shared
network-wide component** — the direction "all sensors err in the same direction" carries
2.75–21 % of its variance, depending on the dataset. On the road graph alone, the prior can
give that direction only 0.5–1.1 % of its variance, at *any* $\gamma$: the constant
eigenvector is one direction out of $N$, and a road graph has no mechanism to make it
special. A dense weak edge does exactly that and nothing else: it raises the Laplacian
eigenvalues of all *rough* patterns uniformly (they now have to disagree with everybody), while
the constant pattern still has $\lambda = 0$. The graph stays a graph; the prior stays the GAD
prior; the connectivity jumps from Fiedler 0.0003 to ≈0.46.

Step (b): **the heat kernel / GMRF.**

$$
\Sigma_S \;\propto\; (L_\beta + \gamma I)^{-1} \;=\; V\,\mathrm{diag}\!\Big(\frac{1}{\lambda_k + \gamma}\Big)\,V^{\!\top}.
$$

*In words:* a Gaussian whose precision (inverse covariance) is $L + \gamma I$. Its density is
$\propto \exp\!\big(-\tfrac12 x^{\!\top}(L+\gamma I)x\big) = \exp\!\big(-\tfrac12 x^{\!\top}Lx - \tfrac{\gamma}{2}\|x\|^2\big)$:
rough patterns are penalised through $x^{\!\top}Lx$, and $\gamma$ is a leak that keeps the
constant pattern (which $L$ cannot see) finite. In the eigenbasis it is diagonal: the pattern
with graph frequency $\lambda_k$ gets variance $1/(\lambda_k + \gamma)$. Small $\lambda$ (smooth,
network-wide) → large variance; large $\lambda$ (rough, sensor-vs-neighbour) → small variance.
$\gamma$ is the ratio knob: at $\gamma \to \infty$ every pattern gets the same variance
(white noise); at $\gamma \to 0$ the constant pattern dominates everything.

Step (c): **unit diagonal.** Per-sensor normalisation (Section 3) already made every sensor's
variance 1, so the prior must say the same:

$$
\Sigma_S \;\leftarrow\; \mathrm{diag}(\Sigma_S)^{-1/2}\;\Sigma_S\;\mathrm{diag}(\Sigma_S)^{-1/2}
\qquad\text{(every } (\Sigma_S)_{ii} = 1\text{; } \gamma \text{ now shapes correlation only).}
$$

Then $\Sigma_S = V_S\,\mathrm{diag}(\mu^S_1,\dots,\mu^S_N)\,V_S^{\!\top}$.

#### 4.2.3 Choosing $(\beta,\gamma)$: match the spread of the network total

We do **not** pick $\beta,\gamma$ by likelihood (a likelihood over $N$ directions is almost blind
to one of them) and not by hand. We pick them so that the prior gets one physically meaningful
number right — the size of the **network-total error** at each horizon, on the training set:

$$
T_h \;=\; \sum_{i=1}^{N} R_{i,h} \;=\; \sum_i \sigma_i\, r_{i,h},
\qquad
\text{measured: } s^{\text{data}}_h = \mathrm{std}(T_h),\qquad
\text{prior-implied: } s^{\text{prior}}_h = \sqrt{\sigma^{\!\top} C_h\, \sigma},\ \ C_h = V_S\,\mathrm{diag}\!\big(\mathrm{var}_{k,h}\big)V_S^{\!\top}.
$$

Over the grid $\beta \in \{0.003, 0.01, 0.03, 0.1, 0.3\}$ and 40 log-spaced
$\gamma \in [10^{-3}, 3]$ we take the pair whose ratio $\overline{s^{\text{prior}}_h / s^{\text{data}}_h}$
is closest to 1. Fresh residuals, today:

| dataset | $\beta$ | $\gamma$ | ratio prior/data | $\theta$ range | derived floor $c_{lo}$ | reverse steps |
|---|---|---|---|---|---|---|
| pems08s | 0.03 | 0.208 | 1.005 | [0.80, 166] | 2.5e-5 | 88 |
| pems04f | 0.003 | 0.0745 | 1.002 | [0.80, 110] | 2.1e-5 | 89 |
| pems04s | 0.01 | 0.169 | 1.020 | [0.80, 219] | 2.1e-5 | 89 |
| metr-la | 0.01 | 0.138 | 1.015 | [0.80, 117] | 2.3e-5 | 88 |
| pems08f | 0.003 | 0.112 | 0.999 | [0.80, 94] | 2.9e-5 | 87 |
| pems-bay | *from the run log when its cell starts* | | | | | |
| iso (all) | — | — | 0.31–0.46 (cannot be matched) | [0.8, 0.8] | 8.1e-4 | 63 |

The last row is the point: white noise gives the total a spread of $\sqrt{N}\,\bar\sigma$
whatever you do, and the measured spread is 2–3× larger because errors are correlated.

**Why the total is where correlation shows.** For $N$ sensors with equal error variance $v$
and pairwise correlation $\rho$,

$$
\mathrm{Var}\Big(\sum_i r_i\Big) \;=\; N v \;+\; N(N-1)\,\rho\, v .
$$

With $N = 170$ and $\rho = 0.1$: $170v + 2873v \approx 18\times$ the independent value, i.e.
the standard deviation is $\approx 4.2\times$ too small under independence. That is not a
subtle effect; it is the whole difference between "the network total will be 100 ± 10 " and
"100 ± 42".

#### 4.2.4 Putting the two factors together: the per-mode precision $\theta$

In the joint eigenbasis $V_S \otimes V_T$ the covariance is diagonal, and so is its inverse
(the **precision**):

$$
\theta_{k,h} \;=\; \frac{1}{\mu^S_k\,\mu^T_h}, \qquad \text{stationary variance of mode }(k,h) = \frac{\sigma^2}{\theta_{k,h}} .
$$

Two rescalings, both exact reparameterisations: (i) $\theta \leftarrow \theta \cdot 0.8/\theta_{\min}$
so the slowest mode sits at $\theta_{\min} = \gamma_{\text{GAD}} = 0.8$ on every dataset (the
schedule's $c_0 = 7$ then mixes it everywhere, $2\cdot 7\cdot 0.8 = 11.2$), with $\sigma^2$
absorbing the factor so nothing about the covariance changes; (ii) $\sigma^2$ set so that
the *mean* stationary variance equals the residual's own $\text{mean}(r^2)$ — **the same total
noise energy as the isotropic arm**. So the arms differ in *shape only*.

A mode is one pattern "sensor-eigenvector $k$ × horizon-eigenvector $h$"; e.g. mode
$(1, 1)$ = "every sensor, every horizon, the same sign" (the network-wide common drift), mode
$(N, H)$ = "checkerboard across roads that flips every 5 minutes" (pure noise). The former has
$\theta = 0.8$ (large variance, slow), the latter $\theta \approx 94$–$219$ (tiny variance, fast).
For iso every mode has $\theta = 0.8$.

---

## 5. Step 4 — the forward process: heat flow on the product graph

The forward SDE (GAD, Rozada et al. 2025) is

$$
\mathrm d r \;=\; -\,c(t)\,Q\,r\,\mathrm dt \;+\; \sqrt{2\,c(t)}\;\mathrm dW, \qquad Q = \Sigma^{-1} .
$$

*In words:* the residual is pulled toward zero along the directions the prior considers
unlikely (strongly, because $Q$ is large there) and only weakly along the directions it
considers likely, while isotropic noise is injected everywhere. Its stationary distribution is
exactly $\mathcal N(0, \Sigma)$ — run it long enough and every residual becomes a draw from
the prior.

**This is the heat equation on the graph.** Drop the noise and take $Q = L + \gamma I$: the
drift is $-c\,L\,r - c\gamma r$, and $\dot r = -L r$ *is* heat diffusion on a graph (each node
moves toward the average of its neighbours). Rough patterns (large $\lambda$) equalise fast,
smooth patterns slowly, the constant pattern only through the leak $\gamma$. The Kronecker
version is heat flow on the **product** of the sensor graph and the horizon axis:
$Q = \Sigma_T^{-1}\otimes\Sigma_S^{-1}$, where $\Sigma_T^{-1}$ plays the role of a temporal
Laplacian (penalising error patterns that change unnaturally fast along the horizon).

**Closed form per mode.** Rotate into the eigenbasis, $C = V_S^{\!\top} r\, V_T$. Each mode is an
independent Ornstein–Uhlenbeck process and, with the *accumulated heat*
$\bar c(t) = \int_0^t c(s)\,\mathrm ds$,

$$
C_{k,h}(t) \;\Big|\; C_{k,h}(0) \;\sim\; \mathcal N\!\Big(\; e^{-\theta_{k,h}\bar c(t)}\; C_{k,h}(0),\;\;
\sigma^2\,\frac{1 - e^{-2\theta_{k,h}\bar c(t)}}{\theta_{k,h}} \Big).
$$

*Read it:* the signal in mode $(k,h)$ is multiplied by $e^{-\theta\bar c}$ (kept fraction) and
noise of the mode's own stationary variance times $(1-e^{-2\theta\bar c})$ is added. A mode is
"half gone" at $\bar c \approx 0.7/\theta$: the checkerboard mode ($\theta \approx 200$) is
erased at $\bar c \approx 0.0035$, the common-drift mode ($\theta = 0.8$) at $\bar c \approx 0.9$.
**The forward process erases fine detail first and the big shared patterns last; the
reverse process therefore *creates* the big shared patterns first and fills in detail last.**
For iso every mode is erased at the same $\bar c$ — there is no order.

**The schedule (FCPS, GAD's exact values).** $c_{\min} = 0.1$, $c_0 = 7$, $\alpha = 4$, $T = 1$:

$$
c(t) = 0.1 + 34.5\,t^{4}, \qquad \bar c(t) = 0.1\,t + 6.9\,t^{5}, \qquad \bar c(1) = c_0 = 7 .
$$

At $t = 1$ the slowest mode keeps $e^{-0.8\cdot 7} = e^{-5.6} \approx 0.4\%$ of its amplitude:
everything is mixed. Because the whole process depends on $(\bar c, \theta)$ only, we index
everything by the noise level $\bar c$ rather than $t$ and sample/step it **log-uniformly** so
every mode's transition ($\bar c\,\theta \sim 1$) is covered evenly across the 4–5.5 decades
between the fastest and slowest mode.

**The floor.** Training draws and the reverse grid run from $c_0$ down to a floor $c_{lo}$.
The old fixed $c_{lo} = 0.02$ is fine for iso (3.1 % of the prior's variance still un-erased
there) but *terrible* for the structured prior: on the fresh matched priors it would leave
**43–59 %** un-denoised (every mode with $\theta \gtrsim 50$ is already pure noise at
$\bar c = 0.02$). So the floor is **derived per prior**:

$$
f(\bar c) \;=\; \frac{\sum_m \mathrm{var}_m\,\big(1 - e^{-2\theta_m \bar c}\big)}{\sum_m \mathrm{var}_m}, \qquad
c_{lo} : \; f(c_{lo}) = 1.3\times10^{-3}
$$

(the fidelity iso gets at its historical $8\times10^{-4}$) — iso: $8.1\times10^{-4}$, kron:
$\approx 2\times10^{-5}$. Both arms are trained and sampled to the **same** fidelity.

---

## 6. Step 5 — training the denoiser

For each training window and each optimiser step:

1. draw a noise level $\bar c \sim \text{log-uniform}[c_{lo}, c_0]$ (one per window),
2. corrupt: $r_{\bar c} = V_S\big(e^{-\theta\bar c}\odot C_0 + \sqrt{\sigma^2(1-e^{-2\theta\bar c})/\theta}\odot Z\big)V_T^{\!\top}$, $Z$ white,
3. predict the clean residual: $\hat r_0 = f_\phi\big(r_{\bar c},\ \tau(\bar c),\ [\,X_z \,|\, \hat Y_z\,]\big)$,
4. loss = masked MSE weighted by GAD's SNR weight,

$$
\mathcal L \;=\; \frac{\sum_{i,h} M_{i,h}\, w\,(\hat r_{0,ih} - r_{0,ih})^2}{\sum_{i,h} M_{i,h}\, w}, \qquad
w = \frac{1}{1 + 2\,c(t)} ,
$$

so the high-noise draws (where the best possible error is the full signal variance) do not
swamp the gradient.

**The denoiser** $f_\phi$ is deliberately small and identical in every cell — **17 305
parameters** — so that nothing but the prior can explain a difference between the arms
(`GCNNDenoiser`): 3 layers of

$$
x \leftarrow \mathrm{ReLU}\Big(\mathrm{FiLM}_{\tau}\Big(\sum_{k=0}^{5} S^{k} x\, W_k \;+\; \mathrm{conv1d}_H(x)\Big)\Big),
$$

i.e. a 5-hop graph filter on the normalised road adjacency $S$ (so it can move information
along roads), a depthwise convolution along the horizon, and a feature-wise affine modulation
by the noise-level embedding $\tau$ (64-dim sin/cos). Input channels are
$[\,r_{\bar c}\,|\,X_z\,|\,\hat Y_z\,]$; the output is residual, $\hat r_0 = r_{\bar c} + \text{correction}$.
For the kron arm the adjacency the denoiser sees is the same $A_\beta$ the prior was built on.

**Training loop settings:** AdamW, lr $10^{-4}$, weight decay $10^{-4}$, batch 32, grad-clip
1, 200 epochs, early-stopping patience 40, LR halved on a plateau of 5. **Checkpoint
selection** is on validation CRPS (512 clean windows, $k = 16$ samples, the *fair* estimator,
the corrected sampler at the derived floor and step count) → `best_crps.pt`. The old
selection used $k = 4$ with a biased estimator that rewarded the narrower ensemble by
+25 % — that is one of the four defects that had been quietly penalising the structured arm.

---

## 7. Step 6 — sampling (the reverse process)

Start from a draw of the prior, $r \sim \mathcal N(0,\Sigma)$, and walk a geometric grid of
noise levels $c_0 = 7 \to c_{lo}$ with **16 steps per decade** (iso: 3.9 decades → 63 steps;
kron: 5.5 decades → 88–89 steps; same resolution per decade of noise, so neither arm is
under-resolved). At each node the denoiser gives $\hat r_0$; the next state is drawn per mode
from the exact posterior of the linear OU process. With $c$ the current and $c'$ the next
(lower) level,

$$
\begin{aligned}
a_d &= e^{-\theta(c-c')}, & s_d &= \sigma^2\,\tfrac{1-e^{-2\theta(c-c')}}{\theta} &&\text{(transition } c'\to c\text{)}\\
a' &= e^{-\theta c'}, & s' &= \sigma^2\,\tfrac{1-e^{-2\theta c'}}{\theta} &&\text{(marginal at } c'\text{)}\\
v &= \frac{s_d\, s'}{a_d^{2} s' + s_d}, & \text{mean} &= v\Big(\frac{a_d\,x_c}{s_d} + \frac{a'\,\hat x_0}{s'}\Big) &&\text{(DDPM posterior, exact mean)}
\end{aligned}
$$

$$
x_{c'} \;=\; \text{mean} + \sqrt{\text{var}}\; z, \qquad
\text{var} \;=\; v \;+\; \Big(\frac{v\,a'}{s'}\Big)^{2}\,\sigma_p^{2}, \qquad
\sigma_p^{2} = \Big(\frac{a_c^{2}}{s_c} + \frac{\theta}{\sigma^{2}}\Big)^{-1} .
$$

*In words:* the standard DDPM step uses $\text{var} = v$, which is the variance of the step
*if $\hat x_0$ were exactly right* (DDPM's $\tilde\beta$ lower bound). We add the second term:
the uncertainty about $x_0$ given the current state under the prior, $\sigma_p^2$, pushed
through the mean. For white noise the two are nearly equal; for the slow modes of a stiff
prior the uncorrected step has **up to half** the correct variance and the deficit compounds
over the chain (the un-corrected sampler needs ~800 steps to reach what the corrected one
reaches at 50). The correction reproduces the closed-form reverse transition to 6 decimals
(`tests/test_sampler_correction.py`). The output is the denoiser's $\hat r_0$ at the last node,
de-normalised: $\hat Y^{(s)} = \hat Y + \sigma_i\,\hat r_0^{(s)}$, for $s = 1,\dots,K$ samples.

---

## 8. Step 7 — what we measure

$K = 100$ samples per test window, $y = 0$ entries masked (the PEMS convention), all six
datasets, four diffusion seeds per arm on top of the shared backbone.

**Per point** (each sensor × horizon separately):

- MAE / RMSE / MAPE of the sample mean;
- normalised CRPS — the number the papers report (CSDI / DiffSTG / SpecSTG convention),

$$
\mathrm{CRPS}_{\text{norm}} \;=\; \frac{1}{19}\sum_{q\in\{0.05,\dots,0.95\}} \frac{2\sum_{\text{pts}} \big|(\hat y_q - y)\,\big(\mathbb 1[y\le\hat y_q] - q\big)\big|}{\sum_{\text{pts}}|y|} ;
$$

- raw-unit **fair** CRPS, $\mathrm{CRPS} = \mathbb E|X - y| - \tfrac12\,\mathbb E|X - X'|$ with the pairwise term normalised by $K(K-1)$ (the $1/K^2$ form is biased upward by $\mathbb E|X-X'|/2K$ — proportional to spread, i.e. it *rewards under-dispersion*);
- 90 % interval coverage and Winkler score, W1 and energy distance per horizon, PIT histograms and reliability curves for the calibration figures.

**Joint / aggregate** — the metrics where the prior can show, computed from the *same* samples:

$$
\text{network total per horizon: } T_h = \sum_i y_{i,h}, \qquad
\text{cumulative per sensor: } U_i = \sum_h y_{i,h},
$$

each scored with fair CRPS, 90 % coverage and the **spread ratio**
$\sqrt{\mathrm{Var}_{\text{samples}} / \text{squared error of the mean}}$ (1 = calibrated,
< 1 = over-confident). Keys `joint_tot_*`, `joint_cum_*` in `metrics.json`.

**What to expect, and why.** Per point, all three (a plain Gaussian on the residual, iso,
kron) land within ≈1 % of each other because the backbone has already won that metric
(GWN + Gaussian, no diffusion, already beats every published number, 0.0681 vs 0.0723 on
pems04f). On the total, iso's spread ratio is ≈0.4 *by construction* (Section 4.2.3) and
its 90 % interval covers ≈50 %; the matched prior swapped into an old checkpoint already gave
0.96–1.12; DiffSTG-1M (58× more parameters) gets 1.12 with coverage 0.30. The campaign's
question is whether a *trained* denoiser keeps the prior's correlation in its samples; the
oracle tests say yes, and the un-corrected sampler was the reason it did not before.

---

## 9. The campaign as queued

| stage | what | where | status |
|---|---|---|---|
| raw → processed | `build_datasets.py --all`, content-verified identical (≤1.5e-5) to the old pipeline's inputs | Mac (CPU) | done |
| backbones | 6 × GWN seed 42, residual caches | DAIC array 285490 | 5/6 done, pems-bay running |
| diffusion | 6 datasets × {iso, kron} × seeds {42, 1, 2, 3} = 48 cells, 2 per job | DAIC `20260907_matched` | next |
| tables/figures | cross-seed aggregation, per-point SOTA table with the Gaussian row, three-way joint table (DiffSTG / iso / kron), calibration figures | Mac (CPU) | after sync |

Fixed per cell: epochs 200, patience 40, lr 1e-4, batch 32, GCNN 17 305 params, norm pernode,
`--cond-past`, $c_{\min}=0.1$, $c_0=7$, $\alpha=4$, $\rho=0.1$, val 512 windows / $k=16$, test $k=100$.
Derived per cell from the fresh train residuals and written to `config.json`: iso $\sigma^2$;
kron $\Sigma_T$, $(\beta,\gamma)$ and the match ratio; $c_{lo}$ and the reverse step count for both.

---

## 10. Symbols

| symbol | meaning |
|---|---|
| $N, H, P$ | sensors, forecast horizon (12 steps = 1 h), history length (12) |
| $Y, \hat Y, R = Y-\hat Y$ | truth, backbone forecast, residual $(N\times H)$ |
| $M$ | observed mask ($y\neq 0$) |
| $\sigma_i$, $r = R/\sigma_i$ | per-sensor residual scale; residual in unit space |
| $A, L, \lambda_k, V$ | adjacency, normalised Laplacian, its eigenvalues (graph frequencies) and eigenvectors |
| $\beta, \bar w$ | global-edge weight relative to the mean road edge |
| $\gamma$ | Laplacian shift / leak in $(L+\gamma I)^{-1}$ (controls smooth-vs-rough ratio) |
| $\Sigma_S, \Sigma_T$ | spatial (unit-diagonal heat kernel) and temporal (horizon variance × correlation) factors |
| $\delta_h, R_H, \rho$ | per-horizon variance, horizon correlation, temporal ridge (0.1) |
| $\mu^S_k, \mu^T_h$ | eigenvalues of $\Sigma_S$, $\Sigma_T$ |
| $\theta_{k,h} = 1/(\mu^S_k\mu^T_h)$ | per-mode precision; mode variance $\sigma^2/\theta$ |
| $c(t), \bar c(t), c_0, c_{lo}$ | schedule rate, accumulated heat, top of the range (7), derived floor |
| $f(\bar c)$ | variance-weighted fraction of the prior already injected at $\bar c$ |
| $\tau$ | normalised log-noise-level the denoiser is conditioned on |
| $K$ | ensemble size (16 for selection, 100 for test) |
| $T_h, U_i$ | network total per horizon, cumulative per sensor (the joint targets) |

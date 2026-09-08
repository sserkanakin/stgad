# The method — one reference for writing the methodology

*The only method document. Everything here is what the code does in the `20260907_matched`
campaign (`src/diffusion/{prior,forward_fcps,reverse,losses,denoiser}.py`, `scripts/{train,eval}_diffusion.py`).
Every formula has a one-line reading under it. The novelty is the prior (Section 3); the rest is
the machinery it needs to be used fairly.*

---

## 1. Notation

| symbol | meaning |
|---|---|
| $N$, $H$, $P$ | sensors, forecast horizon (12 steps of 5 min), history length (12) |
| $X \in \mathbb R^{N\times P}$, $Y \in \mathbb R^{N\times H}$ | observed past, future to predict |
| $\hat Y = \mathrm{GWN}(X)$ | backbone point forecast |
| $R = Y - \hat Y$ | the **residual field** $(N\times H)$; $M\in\{0,1\}^{N\times H}$ its observed-mask |
| $\sigma_i$ | training std of sensor $i$'s residual; $r = R/\sigma_i$ the residual in unit space |
| $A$, $L$ | road adjacency; normalised Laplacian $L = I - D^{-1/2}AD^{-1/2}$, eigenpairs $(\lambda_k, v_k)$, $0=\lambda_1\le\dots\le\lambda_N\le 2$ |
| $\Sigma$ | the prior covariance: $\sigma^2 I$ (isotropic) or $\Sigma_T\otimes\Sigma_S$ (Kronecker) |
| $\theta_{k,h}$ | precision of mode $(k,h)$ in the prior's eigenbasis; mode variance $\sigma^2/\theta_{k,h}$ |
| $\bar c$ | accumulated noise ("heat") of the forward process; $c_0 = 7$ its top, $c_{lo}$ the floor |
| $K$ | ensemble size: 16 for checkpoint selection, 100 for the test evaluation |

---

## 2. Problem and pipeline

A deterministic backbone gives a good point forecast $\hat Y$. Its error $R = Y-\hat Y$ is not
noise: it is correlated along the horizon, along the road graph, and network-wide. We want the
*distribution* of $R$ given what is known, and in particular the distribution of **aggregates**
(network total per horizon, cumulative per sensor) that operators act on.

Pipeline: $X \xrightarrow{\ \mathrm{GWN}\ } \hat Y \to R = Y-\hat Y \to r = R/\sigma_i
\xrightarrow{\ \text{diffusion with prior }\Sigma\ } \{\hat r^{(s)}\}_{s=1}^K \to \hat Y^{(s)} = \hat Y + \sigma_i\hat r^{(s)}$.

Per-sensor scale:
$$
\sigma_i = \mathrm{std}\{R_{i,h}: \text{train windows, observed}\},\qquad r_{i,h} = R_{i,h}/\sigma_i .
$$
*Reading: every sensor has unit residual variance on train, so the prior only has to describe correlation, not scale.*

The backbone (Graph WaveNet, ≈309k parameters, 100 epochs, best epoch by validation MAE, seed 42)
is trained once per dataset and shared by every diffusion cell: **both arms see the same residuals.**

---

## 3. The prior — the contribution

A diffusion model corrupts data toward a Gaussian $\mathcal N(0,\Sigma)$ and learns to reverse it.
Standard practice takes $\Sigma = \sigma^2 I$ (white noise). Our claim: **for forecast residuals the
right $\Sigma$ is a Kronecker covariance whose spatial factor is the road graph's heat kernel plus
one weak global edge, with two scalars set so the prior's spread on the network total is right.**

### 3.1 Isotropic prior (the control)
$$
\Sigma_{\text{iso}} = \sigma^2 I_{NH},\qquad \sigma^2 = \mathrm{mean}\{r_{i,h}^2 : \text{observed, train}\}.
$$
*Reading: independent noise of the same size at every sensor and horizon. Same total noise energy as the structured prior, no structure.*

### 3.2 Kronecker structure
For $r\in\mathbb R^{N\times H}$, with $\mathrm{vec}$ horizon-major (entry $hN+i \leftrightarrow r_{i,h}$):
$$
\mathrm{Cov}(r_{i,h}, r_{j,h'}) = (\Sigma_T)_{hh'}\,(\Sigma_S)_{ij},\qquad \Sigma = \Sigma_T\otimes\Sigma_S .
$$
*Reading: correlation between two entries = (how their horizons relate) × (how their sensors relate). The $NH\times NH$ matrix is never formed; only the two factors and their eigendecompositions are.*

Toy: two neighbouring sensors (corr 0.5), two horizons (corr 0.9, second has twice the variance):
$$
\Sigma_S=\begin{pmatrix}1&.5\\.5&1\end{pmatrix},\ 
\Sigma_T=\begin{pmatrix}1&.9\sqrt2\\ .9\sqrt2&2\end{pmatrix},\ 
\Sigma_T\otimes\Sigma_S=\begin{pmatrix}1&.5&1.27&.64\\ .5&1&.64&1.27\\ 1.27&.64&2&1\\ .64&1.27&1&2\end{pmatrix}.
$$
Measured justification: the empirical residual covariance is 91–94 % captured by a single Kronecker product (nearest-Kronecker-product test), on every dataset.

### 3.3 Temporal factor $\Sigma_T$ — fitted from the training residual
$$
\delta_h = \mathrm{mean}\{r_{i,h}^2\},\qquad (R_H)_{hh'} = \mathrm{corr}(r_{\cdot,h}, r_{\cdot,h'}),\qquad
\Sigma_T = D_H^{1/2} R_H D_H^{1/2},\ D_H=\mathrm{diag}(\delta_h).
$$
*Reading: the diagonal is the error variance at each horizon (grows 1.5–2× over the hour); the off-diagonal is the Pearson correlation between horizons (adjacent ≈ 0.9).*
Eigendecompose $\Sigma_T = V_T\,\mathrm{diag}(\mu^T_h)\,V_T^{\!\top}$ and apply a trace-preserving ridge toward its mean, $\mu^T_h \leftarrow (1-\rho)\mu^T_h + \rho\,\bar\mu^T$, $\rho = 0.1$.

### 3.4 Spatial factor $\Sigma_S$ — the road graph's heat kernel plus a global edge

**(a) The graph.** Self-loops removed; disconnected components joined to the giant component by one
edge each (between the two most signal-correlated sensors, median edge weight) so every graph is
connected. Then the **global edge**:
$$
A_\beta = A_{\text{road}} + \beta\,\bar w\,(\mathbf 1\mathbf 1^{\!\top} - I),\qquad \bar w = \text{mean road-edge weight},\ \beta\in[0.003, 0.03].
$$
*Reading: every pair of sensors gets a link worth 0.3–3 % of a typical road link.*

**(b) The heat kernel / GMRF.** With $L_\beta$ the normalised Laplacian of $A_\beta$:
$$
\Sigma_S \propto (L_\beta + \gamma I)^{-1} = V\,\mathrm{diag}\!\Big(\tfrac{1}{\lambda_k+\gamma}\Big)V^{\!\top}.
$$
*Reading: a Gaussian whose density is $\propto\exp(-\tfrac12 x^{\!\top}L x - \tfrac\gamma2\|x\|^2)$: patterns that are rough across roads ($x^{\!\top}Lx$ large) are unlikely; $\gamma$ is a leak that keeps the constant pattern ($\lambda=0$) finite. Pattern with graph frequency $\lambda_k$ gets variance $1/(\lambda_k+\gamma)$.*

**(c) Unit diagonal.** Per-sensor scaling already made every sensor's variance 1, so
$$
\Sigma_S \leftarrow \mathrm{diag}(\Sigma_S)^{-1/2}\,\Sigma_S\,\mathrm{diag}(\Sigma_S)^{-1/2},
$$
*Reading: $\Sigma_S$ is a correlation matrix; $\gamma$ and $\beta$ shape correlation only.* Eigendecompose $\Sigma_S = V_S\,\mathrm{diag}(\mu^S_k)\,V_S^{\!\top}$.

### 3.5 Why the global edge (the measured argument)
The residual has a **shared network-wide component**: the direction "every sensor errs the same
way" carries 2.5–6.5 % of the variance (up to 21 % on speed data), whereas independence gives $1/N \approx 0.3$–0.6 %.
Consequently the sd of the network total is 1.6–2.8× what independent errors would give.
The road graph alone cannot represent this at any $\gamma$: lowering $\gamma$ inflates *all* smooth
patterns together, so by the time the total is right the neighbour correlation is 3× too strong
(pems08f: 0.49 vs 0.14 measured). The global edge raises the frequency of every non-constant
pattern by about the same amount while the constant pattern stays at $\lambda=0$ — it lifts the
network-wide share *specifically*. Two knobs, two targets: $\beta$ for the network-wide share, $\gamma$ for the neighbourhood.

### 3.6 Choosing $(\beta,\gamma)$: match the spread of the network total
Network total of the raw residual and its measured / prior-implied sd:
$$
T_h = \sum_i \sigma_i r_{i,h},\qquad s^{\text{data}}_h = \mathrm{std}(T_h),\qquad
s^{\text{prior}}_h = \sqrt{\sigma^{\!\top} C_h\,\sigma},\quad C_h = V_S\,\mathrm{diag}(\mathrm{var}_{k,h})\,V_S^{\!\top},
$$
$$
(\beta,\gamma) = \arg\min_{\beta\in\{.003,.01,.03,.1,.3\},\ \gamma\in[10^{-3},3]}\ \Big|\,\mathrm{mean}_h\, s^{\text{prior}}_h/s^{\text{data}}_h - 1\,\Big| .
$$
*Reading: pick the pair whose implied uncertainty of the whole network's error equals the measured one. Training residuals only; no likelihood (a likelihood over $N$ directions is nearly blind to one of them).*
Result on the fresh residuals: ratio 0.999–1.020 on train, 0.91–1.02 on test, and — not matched, measured — the prior is also right at every intermediate scale (1-hop neighbourhoods, spectral clusters of 64/16/4), where iso and the road-only prior collapse to 0.3–0.5.

### 3.7 Modes, precision, scale
In the joint basis $V_S\otimes V_T$ the prior is diagonal with precision
$$
\theta_{k,h} = \frac{1}{\mu^S_k\,\mu^T_h},\qquad \text{mode variance} = \frac{\sigma^2}{\theta_{k,h}} .
$$
Two exact reparameterisations: $\theta \leftarrow \theta\cdot 0.8/\theta_{\min}$ (slowest mode at GAD's $\theta_{\min}=\gamma_{\rm GAD}=0.8$, so $2c_0\theta_{\min}=11.2$ on every dataset), and $\sigma^2$ set so the *mean* mode variance equals $\mathrm{mean}(r^2)$ — **the same total noise energy as the isotropic arm**. Mode $(1,1)$ = "every sensor, every horizon, same sign" ($\theta=0.8$, large variance, slow); mode $(N,H)$ = a road-checkerboard flipping every 5 min ($\theta\approx 94$–$629$, tiny variance, fast). For iso every mode has $\theta=0.8$.

### 3.8 What the prior does *not* claim
It is Gaussian while the residual is heavy-tailed (kurtosis 5–8): the prior is the **noise process**, the denoiser supplies the tails. It over-correlates road neighbours (0.33 vs 0.14) — the honest cost of one-moment matching. It is fitted on train; the test period can be harder (pems08s: 2.4× residual variance), which caps every calibrated spread there.

---

## 4. The forward process (heat flow on the product graph)
$$
\mathrm d r = -\,c(t)\,\Sigma^{-1} r\,\mathrm dt + \sqrt{2c(t)}\,\mathrm dW,\qquad
\text{stationary law } \mathcal N(0,\Sigma).
$$
*Reading: the residual is pulled toward zero along the directions the prior finds unlikely and weakly along the likely ones, while white noise is injected — the graph heat equation with a leak, run on the product of the sensor graph and the horizon axis.*
Per mode, with $\bar c(t)=\int_0^t c$:
$$
C_{k,h}(t)\,\big|\,C_{k,h}(0) \sim \mathcal N\!\Big(e^{-\theta_{k,h}\bar c}\,C_{k,h}(0),\ \sigma^2\,\tfrac{1-e^{-2\theta_{k,h}\bar c}}{\theta_{k,h}}\Big),\qquad C = V_S^{\!\top} r\,V_T .
$$
*Reading: a mode is half erased at $\bar c\approx 0.7/\theta$: fine road detail first ($\theta\sim 200$: $\bar c\sim 0.004$), the network-wide pattern last ($\theta=0.8$: $\bar c\sim 0.9$). The reverse creates the big shared patterns first and fills in detail last. For iso there is no order.*

Schedule (GAD's FCPS, verbatim): $c_{\min}=0.1$, $c_0=7$, $\alpha=4$, $T=1$:
$$
c(t) = 0.1 + 34.5\,t^4,\qquad \bar c(t) = 0.1\,t + 6.9\,t^5,\qquad \bar c(1)=7 .
$$
Noise levels are drawn/stepped **log-uniformly in $\bar c$** so every mode's transition is covered evenly.

**Derived floor.** Un-denoised fraction and floor:
$$
f(\bar c) = \frac{\sum_m \mathrm{var}_m\,(1-e^{-2\theta_m\bar c})}{\sum_m \mathrm{var}_m},\qquad c_{lo}:\ f(c_{lo}) = 1.3\times10^{-3}.
$$
*Reading: both arms are trained and sampled to the same fidelity. iso: $c_{lo}=8.1\times10^{-4}$; kron: $0.7$–$2.9\times10^{-5}$. The old fixed floor 0.02 left 3 % of iso but 43–62 % of the matched prior un-denoised.*

---

## 5. Denoiser and training
Per training window: draw $\bar c\sim\mathrm{logU}[c_{lo}, c_0]$; corrupt $r\to r_{\bar c}$ by the closed form; predict
$$
\hat r_0 = f_\phi\big(r_{\bar c},\ \tau(\bar c),\ [\,X_z\,|\,\hat Y_z\,]\big),
$$
*conditioning = z-scored observed history and z-scored backbone forecast, concatenated along time $(N, 24)$.*
Loss (GAD's SNR-weighted masked MSE on the clean residual):
$$
\mathcal L = \frac{\sum_{i,h} M_{i,h}\,w\,(\hat r_{0,ih}-r_{0,ih})^2}{\sum_{i,h} M_{i,h}\,w},\qquad w=\frac{1}{1+2c(t)} .
$$
Denoiser $f_\phi$: **17,305 parameters, identical in every cell** — 3 layers of
$x\leftarrow\mathrm{ReLU}(\mathrm{FiLM}_\tau(\sum_{k=0}^{5}S^k x W_k + \mathrm{conv1d}_H(x)))$ on the normalised adjacency $S$ (of $A_\beta$ for the kron arm), residual output $\hat r_0 = r_{\bar c}+$ correction.
Optimiser AdamW $10^{-4}$, wd $10^{-4}$, batch 32, ≤200 epochs, patience 40, EMA. **Checkpoint = best validation CRPS** (512 windows, $K=16$, fair estimator, the sampler of Section 6 at the derived floor and steps).

---

## 6. Sampling — exact per-mode ancestral step
Start $r\sim\mathcal N(0,\Sigma)$; geometric grid $c_0\to c_{lo}$ with 16 steps per decade (iso 63, kron 87–96). With $c$ current, $c'$ next, per mode:
$$
a_d=e^{-\theta(c-c')},\ s_d=\sigma^2\tfrac{1-e^{-2\theta(c-c')}}{\theta};\quad a'=e^{-\theta c'},\ s'=\sigma^2\tfrac{1-e^{-2\theta c'}}{\theta};\quad
v=\frac{s_d s'}{a_d^2 s'+s_d},\ \ \text{mean}=v\Big(\frac{a_d x_c}{s_d}+\frac{a'\hat x_0}{s'}\Big)
$$
$$
x_{c'} = \text{mean}+\sqrt{\text{var}}\,z,\qquad \text{var} = v + \Big(\frac{v a'}{s'}\Big)^{2}\sigma_p^2,\qquad
\sigma_p^2=\Big(\frac{a_c^2}{s_c}+\frac{\theta}{\sigma^2}\Big)^{-1}.
$$
*Reading: the DDPM posterior mean, plus the variance of $x_0$ given the current state under the prior pushed through the mean. Without the second term (DDPM's $\tilde\beta$ choice) the step variance is up to 2× too small for slow modes and the deficit compounds. Exact to $10^{-6}$ against the closed form; no stiffness limit, unlike Euler–Maruyama, which needs $n\gtrsim\theta_{\max}c_0$ (660–4,400 steps for kron).*

**The DC rule (from the chain diagnostic).** For the one mode $k^\star=\arg\max_k|v_k^{\!\top}\mathbf 1|$ (the network-wide pattern), $\hat x_0$ is taken from the prior's own optimal filter instead of the network:
$$
\hat C_{0,k^\star,h} = \sigma_p^2\,\frac{a_c}{s_c}\,C_{k^\star,h}\qquad\text{(GAD Prop. 1 for that mode)} .
$$
*Reading: the learned graph filter handles every graph frequency except DC; on DC its estimate is worse than the prior's (its ensemble-mean total is worse than predicting zero) and its spread is 0.58× — with this rule the total is calibrated (1.00×). For iso the mode carries $1/N$ of the variance and the rule changes nothing; it is applied identically to both arms.*

---

## 7. Evaluation (standard names)
$K=100$ samples per test window, $y=0$ masked, from `best_crps.pt`.
- Per point: normalised quantile CRPS (CSDI/DiffSTG convention), MAE, RMSE, PICP (90 % coverage), MSIS/Winkler, energy score.
- Joint: **CRPS_sum** on the network total $T_h=\sum_i y_{i,h}$ (TimeGrad/CSDI's multivariate metric) and on the cumulative per sensor $U_i=\sum_h y_{i,h}$, each with PICP and the spread ratio $\sqrt{\mathrm{Var}_{\rm samples}/\text{err}^2}$.
- GAD-style: aMMD of graph-signal statistics (QV, SC, DC) of samples vs test residuals; quality vs number of reverse steps.
Fair CRPS estimator: $\mathrm{CRPS}=\mathbb E|X-y|-\tfrac12\mathbb E|X-X'|$ with the pairwise term over $K(K-1)$.

---

## 8. What changed versus the earlier pipeline, and why the result flipped
| change | before | now | effect |
|---|---|---|---|
| spatial prior | road graph, $\gamma=0.8$ fixed | road + global edge, $(\beta,\gamma)$ matched on the total | network-scale spread 0.5 → 1.0 (prior level) |
| sampler variance | DDPM $\tilde\beta$ (drops $\sigma_p^2$) | exact per-mode step | up to 2× more variance per step on slow modes |
| noise floor / steps | fixed $c_{lo}=0.02$, 50 steps | derived $f(c_{lo})=1.3\times10^{-3}$, 16 steps/decade | kron no longer 43–62 % un-denoised |
| checkpoint selection | biased CRPS, $K=4$ | fair CRPS, $K=16$ | no reward for under-dispersion |
| metrics | per point only | + CRPS_sum, cumulative, PICP, spread | the axis where the prior acts |
| DC mode | learned | prior's filter (eval-only option) | total coverage ≈0.65 → ≈0.9 |
Same backbone, same denoiser, same data, same test set: kron coverage 0.70–0.77 → 0.85, per-point CRPS −0.4 to −3 %, iso unchanged; joint metrics kron better on 6/6 datasets.

---

## 9. Per-dataset derived numbers (appendix material)
| dataset | $N$ | $\beta$ | $\gamma$ | match ratio | $\theta$ range | $c_{lo}$ | steps |
|---|---|---|---|---|---|---|---|
| pems08f | 170 | 0.003 | 0.112 | 0.999 | [0.8, 94] | 2.9e-5 | 87 |
| pems08s | 170 | 0.03 | 0.208 | 1.005 | [0.8, 166] | 2.5e-5 | 88 |
| pems04f | 307 | 0.003 | 0.075 | 1.002 | [0.8, 110] | 2.1e-5 | 89 |
| pems04s | 307 | 0.01 | 0.169 | 1.020 | [0.8, 219] | 2.1e-5 | 89 |
| metr-la | 207 | 0.01 | 0.138 | 1.015 | [0.8, 117] | 2.3e-5 | 88 |
| pems-bay | 325 | 0.03 | 0.061 | 1.004 | [0.8, 629] | 7.4e-6 | 96 |
| iso (all) | — | — | — | 0.31–0.46 | [0.8, 0.8] | 8.1e-4 | 63 |

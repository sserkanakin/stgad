"""regenerate the three per-dataset val-eval notebooks from one template.

run:  python notebooks/_make_val_eval_notebooks.py
emits: notebooks/val_eval_<dataset>.ipynb  for metrla / pems08f / pemsbay

the notebooks are thin: all machinery lives in notebooks/val_eval_lib.py, which
reuses the production code paths so the numbers match training / eval_diffusion.
keeping this generator means the three stay identical except the dataset name.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

UNIT = {"metrla": "mph", "pemsbay": "mph", "pems08f": "veh"}
OUT = Path(__file__).resolve().parent


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip("\n"))


def build(ds: str) -> nbf.NotebookNode:
    unit = UNIT[ds]
    cells = [
        md(f"""
# {ds} — val diagnostics: reconstruction & generative distribution

Two evals on the **validation** residuals of `{ds}`, comparing the
**Kronecker (structured)** prior against the **isotropic** baseline. All numbers are in
raw units (**{unit}**) — the backbone forecast cancels, so a residual error *is* the
forecast error and we score residuals directly.

1. **Reconstruction** — fully noise the *true* val residuals to `t = T`, then run the
   deterministic reverse sampler back to `t = 0`, and report MAE / MSE.
   *Does the denoiser invert the forward process?* (Oracle-seeded — a sanity floor, not
   generative skill.)
2. **Generative distribution** — draw *pure-prior* samples (no true signal), run the
   full stochastic reverse SDE, and compare the generated residual cloud to the true val
   cloud per horizon (sliced-W1 + energy), plus sample-mean MAE/MSE and the Winkler-90
   interval + empirical coverage. *Does the model generate realistic residuals?*

> The trained denoisers live on DAIC under `results/diffusion/…`. Run this notebook where
> those checkpoints are present (or sync them down); every cell degrades cleanly with a
> message if a checkpoint is missing.
"""),
        md("## Setup & configuration"),
        code(f"""
import sys, pathlib
_here = pathlib.Path.cwd()
ROOT = _here if (_here / "src").exists() else _here.parent     # works from repo root or notebooks/
for _p in (str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import val_eval_lib as ve

# ── config ───────────────────────────────────────────────────────────────────
DATASET = "{ds}"
SEED    = 42            # which trained seed to load (smallest available is used if absent)

# sampling settings — match scripts/eval_diffusion.py defaults; lower for a quick pass
N_STEPS       = 100     # reverse-sampler steps
RECON_WINDOWS = 1024    # val windows for eval 1 (1 trajectory each, cheap)
K             = 50      # prior samples per window for eval 2 point / interval metrics
GEN_WINDOWS   = 512     # val windows for eval 2 point / interval metrics
ETA           = 1.0     # 1.0 = full stochastic reverse SDE (proper interval dispersion)
ALPHA         = 0.1     # 0.1 -> central 90% Winkler interval
DIST_WINDOWS  = 512     # complete-case windows for the W1 / distribution comparison
DIST_SAMPLES  = 4       # prior samples per window for the W1 / distribution cloud

DEVICE = ve.pick_device()
print("device:", DEVICE)
"""),
        md("## Load the trained denoisers\n\nDiscovers checkpoints under `results/diffusion/**` and loads each prior; missing priors are skipped with a message."),
        code("""
runs = ve.load_runs(DATASET, device=DEVICE, seed=SEED)   # {prior: run}; skips missing checkpoints
"""),
        md(f"""
## Eval 1 — Reconstruction (fully noise → denoise)

We take the true val residuals, **forward-noise them to the top of the schedule** (`t = T`,
essentially pure prior noise), then run the **deterministic** reverse sampler (`eta = 0`)
back down to `t = 0`. The reconstruction MAE/MSE (in {unit}) and the MAE-vs-`t` curve tell
us whether the learned reverse process actually undoes the forward noising.
"""),
        code("""
recon = {
    prior: ve.reconstruction_eval(
        run, device=DEVICE, n_steps=N_STEPS, max_windows=RECON_WINDOWS,
        generator=ve.make_generator(DEVICE, SEED))
    for prior, run in runs.items()
}
print("done" if recon else "no checkpoints loaded — run where results/diffusion/ exists.")
(ve.reconstruction_table(recon) if recon else None)
"""),
        code("if recon: ve.plot_reconstruction(recon, DATASET)"),
        md("""
**How to read it.** The left curve should fall as `t → 0`: the input starts as noise
(high MAE) and a working denoiser drives it down to a small reconstruction error. The
right bars give final per-horizon MAE — error grows with horizon (further-ahead residuals
are larger). This eval is *oracle-seeded* (it starts from the true signal's noise), so a
low MAE only confirms the reverse process is **consistent** — it's a floor, not the
model's forecasting skill. A flat/rising curve, or a recon MAE near the data scale, means
sampling is broken.
"""),
        md(f"""
## Eval 2 — Generative distribution (prior → denoise → vs val)

Now we use **no true signal**: we draw pure-prior samples and run the **full stochastic**
reverse SDE (`eta = {{ETA}}`). For the highlighted horizons we overlay the generated
residual distribution on the true val distribution, and per horizon we measure
**sliced-W1 + energy distance** in node-space `R^N`. We also report the K-sample-mean
**MAE/MSE**, the **Winkler-90** interval score and **coverage** — all in {unit}.
""".replace("{{ETA}}", "ETA")),
        code("""
gen = {
    prior: ve.generative_eval(
        run, device=DEVICE, k=K, n_steps=N_STEPS, eta=ETA, alpha=ALPHA,
        max_windows=GEN_WINDOWS, dist_windows=DIST_WINDOWS, dist_samples=DIST_SAMPLES,
        generator=ve.make_generator(DEVICE, SEED))
    for prior, run in runs.items()
}
print("done" if gen else "no checkpoints loaded — run where results/diffusion/ exists.")
(ve.generative_table(gen) if gen else None)
"""),
        code("if gen: ve.plot_generative(gen, DATASET)"),
        md("""
**How to read it.** The histograms should *match in spread and shape* (these residuals
are heavy-tailed, so watch the tails, not just the peak). **W1 / energy** (lower = better)
quantify that match in node-space `R^N`, where the structured prior's spatial correlation
should help and the isotropic baseline — which can match marginals but not the joint
spatial structure — should fall behind. **gen MAE/MSE** is point skill of the sample mean.
**Winkler-90** combines sharpness and calibration (lower = better) and must be read with
**coverage** (target ≈ 0.90): a low Winkler with coverage well below 0.90 just means the
intervals are too narrow.
"""),
        md("""
### Caveats
- **One seed.** This loads `SEED`; the headline thesis numbers aggregate seeds 42 / 43 / 44.
- **Finite K.** With `K` samples the 5% / 95% quantiles (hence Winkler & coverage) are noisy — raise `K` for tighter estimates.
- **Two noise levels.** Eval 1 is deterministic (`eta = 0`, spread from prior seeds only); eval 2 is stochastic (`eta = 1`). Don't cross-compare their dispersions.
- **Val, not test.** These are validation diagnostics; the reported *test* numbers come from `scripts/eval_diffusion.py`.
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    for ds in ("metrla", "pems08f", "pemsbay"):
        path = OUT / f"val_eval_{ds}.ipynb"
        nbf.write(build(ds), path)
        print("wrote", path)


if __name__ == "__main__":
    main()

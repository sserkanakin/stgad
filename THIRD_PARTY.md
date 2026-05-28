# Third-party vendored dependencies

`third_party/` is gitignored; clone the repos below locally to populate it.

## Graph-WaveNet (GWN)

Backbone for residual extraction (phase 1).

```bash
cd third_party
git clone https://github.com/nnzhan/Graph-WaveNet.git
```

Imported by `src/backbones/__init__.py` and used by `scripts/train_backbone.py`.

## DiffSTG (UGnet)

Source of the U-Net denoiser architecture (Stage C).

```bash
cd third_party
git clone --depth 1 https://github.com/wenhaomin/DiffSTG.git
```

Pinned reference commit: `ed2c523022a2490ddb9b78aab641eac53b0274d3` (2024-03-09).

Files we consume:

- `algorithm/diffstg/ugnet.py` — `UGnet`, `ResidualBlock`, `SpatialBlock`,
  `TcnBlock`, `TimeEmbedding`, `MiddleBlock`, etc.
- `algorithm/diffstg/graph_algo.py` — graph adjacency helpers (`asym_adj`).

`src/diffusion/denoiser.py` adapts these — continuous-time `t`, Tweedie R₀
target (not ε), conditioning on (X history, Ŷ backbone) instead of DiffSTG's
masked-imputation framing.

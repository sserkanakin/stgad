"""One-shot diagnostic: trace reverse-SDE magnitudes for pemsbay structured
vs isotropic vs metrla structured. Run locally on CPU/MPS with batch=2, K=4.
"""
from __future__ import annotations
import sys, json, numpy as np, torch, yaml
sys.path.insert(0, '.')

from src.diffusion.score_net import ScoreNet
from src.diffusion.sde import eigendecompose_prior
from src.diffusion.sde_torch import eig_to_torch, sample_stationary, to_spectral
from src.diffusion.prior_fit import build_Q_H, build_Sigma_S
from src.data.adjacency import load_adjacency, compute_normalized_laplacian
from scripts.eval_diffusion import TestDataset, build_eig_isotropic, build_adjacency_torch, _NUM_NODES

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cpu')  # determinism

def load_run(run_id: str):
    import glob
    ckpt_dir = glob.glob(f'results/checkpoints/{run_id}')[0]
    ck = torch.load(f'{ckpt_dir}/best.pt', map_location=device, weights_only=False)
    cfg = ck['config']
    dataset = cfg['dataset']
    prior_type = cfg['prior']

    yaml_p = yaml.safe_load(open(f'results/prior_params/gwn_{dataset}.yaml'))
    target_scale = np.asarray(yaml_p['target_scale'], dtype=np.float32)
    if prior_type == 'structured':
        delta_h = np.asarray(yaml_p['delta_h'])
        Q_H = build_Q_H(delta_h, yaml_p['alpha'], yaml_p['rho1'], yaml_p['rho2'])
        adj = load_adjacency(dataset)
        L = compute_normalized_laplacian(adj)
        eL, vL = np.linalg.eigh(L)
        Sigma_S = build_Sigma_S(yaml_p['beta'], yaml_p['epsilon_s'], eL, vL)
        eig_np = eigendecompose_prior(Q_H, Sigma_S)
    else:
        eig_np = build_eig_isotropic(_NUM_NODES[dataset])
    eig_t = eig_to_torch(eig_np, device=device, dtype=torch.float32)

    adj_t = build_adjacency_torch(dataset, device=device)
    model = ScoreNet(num_sensors=_NUM_NODES[dataset], adj=adj_t,
                     hidden_dim=cfg['hidden_dim'], num_blocks=cfg['num_blocks'],
                     tau_embed_dim=cfg['tau_embed_dim']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    test_ds = TestDataset(dataset, seed=0, target_scale=target_scale)
    return model, eig_t, target_scale, cfg, test_ds, dataset, prior_type


@torch.no_grad()
def trace_reverse(model, eig, x_history, y_pred, n_steps, beta_min, beta_max, tau_eps, dataset, prior_type):
    """Run reverse SDE with a single trajectory and print magnitudes."""
    from src.diffusion.sde_torch import _forward_decay_var, apply_Q, g_squared, from_spectral

    K = 4
    B = x_history.shape[0]
    H, N = eig['s'].shape
    x_history_rep = x_history.unsqueeze(1).expand(B, K, *x_history.shape[1:]).reshape(B*K, *x_history.shape[1:])
    y_pred_rep = y_pred.unsqueeze(1).expand(B, K, H, N).reshape(B*K, H, N)

    s_eig = eig['s']
    print(f"\n=== {dataset} / {prior_type} ===")
    print(f"  eig.s: min={s_eig.min():.3e} median={s_eig.median():.3e} max={s_eig.max():.3e}")

    r = sample_stationary(B*K, eig, dtype=y_pred.dtype)
    print(f"  τ=1.000  r_init                 std_node={r.std():.3f}  max|r|={r.abs().max():.3f}")

    taus = torch.linspace(1.0, tau_eps, n_steps+1)
    log_steps = [0, n_steps//5, n_steps//2, 4*n_steps//5, n_steps-1]
    for k in range(n_steps):
        tau_k = float(taus[k])
        dt = float(taus[k] - taus[k+1])
        tau_batch = torch.full((r.shape[0],), tau_k, dtype=r.dtype, device=r.device)

        r_hat_0 = model(r, tau_batch, x_history_rep, y_pred_rep)

        if k in log_steps:
            decay, var = _forward_decay_var(tau_k, eig, beta_min, beta_max)
            r_hat_t = to_spectral(r_hat_0, eig['U_T'], eig['U_S'])
            mu_t = decay * r_hat_t
            r_t = to_spectral(r, eig['U_T'], eig['U_S'])
            score_t = -(r_t - mu_t) / var
            score = from_spectral(score_t, eig['U_T'], eig['U_S'])
            g2 = float(g_squared(tau_k, beta_min, beta_max))
            Q_r = apply_Q(r, eig)
            drift = -0.5*g2*Q_r - g2*score
            print(f"  τ={tau_k:.3f}  k={k:3d}  std(r)={r.std():.3f}  std(r̂₀)={r_hat_0.std():.3f}  "
                  f"std(score)={score.std():.3f}  std(drift)={drift.std():.3f}  "
                  f"g²={g2:.3f}  var.min={var.min():.3e}")

        # Manual step
        decay, var = _forward_decay_var(tau_k, eig, beta_min, beta_max)
        r_hat_t = to_spectral(r_hat_0, eig['U_T'], eig['U_S'])
        mu_t = decay * r_hat_t
        r_t = to_spectral(r, eig['U_T'], eig['U_S'])
        score_t = -(r_t - mu_t) / var
        score = from_spectral(score_t, eig['U_T'], eig['U_S'])
        g2 = float(g_squared(tau_k, beta_min, beta_max))
        Q_r = apply_Q(r, eig)
        drift = -0.5*g2*Q_r - g2*score
        eps = torch.randn(r.shape, dtype=r.dtype)
        r = r - drift*dt + (g2*dt)**0.5 * eps

    print(f"  τ={float(taus[-1]):.3f}  r_final (standardised)  std={r.std():.3f}  max|r|={r.abs().max():.3f}")
    return r


def main(run_ids):
    for run_id in run_ids:
        model, eig, ts, cfg, test_ds, dataset, prior_type = load_run(run_id)
        # Grab one batch of 2 windows
        b0 = test_ds[0]
        b1 = test_ds[1]
        x_h = torch.stack([b0['x_history'], b1['x_history']], dim=0).to(device)
        y_p = torch.stack([b0['y_pred_std'], b1['y_pred_std']], dim=0).to(device)
        y_true_raw = torch.stack([b0['y_true_raw'], b1['y_true_raw']], dim=0).numpy()
        y_pred_raw = torch.stack([b0['y_pred_raw'], b1['y_pred_raw']], dim=0).numpy()

        n_steps_arg = int(__import__('os').environ.get('DIAG_N_STEPS', '100'))
        r_final = trace_reverse(model, eig, x_h, y_p, n_steps=n_steps_arg,
                                 beta_min=cfg['beta_min'], beta_max=cfg['beta_max'],
                                 tau_eps=1e-3, dataset=dataset, prior_type=prior_type)
        # De-standardise (last K=4 samples for window 0)
        r_np = r_final.numpy().reshape(2, 4, 12, _NUM_NODES[dataset])
        samples_mph = y_pred_raw[:, None, :, :] + r_np * ts.reshape(1, 1, 12, 1)
        per_h_std = samples_mph.std(axis=1).mean(axis=(0, 2))  # (H,)
        print(f"  per-horizon sample std (mph) avg: {per_h_std.mean():.2f}  per-h: {per_h_std.round(2).tolist()}")


if __name__ == '__main__':
    runs = sys.argv[1:] if len(sys.argv) > 1 else [
        'dsm_isotropic_pemsbay_seed0_a15ca06b',
        'dsm_structured_metrla_seed0_36f7a00f',
        'dsm_structured_pemsbay_seed0_09034d3d',
    ]
    main(runs)

# History-conditioned reruns — what changed and what to launch

## Code changes already applied
1. `scripts/extract_residuals.py` now also saves the observed history window
   (`history`, shape (B,N,P), raw units) into each residual `.npz`.
2. `src/data/residual_dataset.py`: with `cond_past=True` the conditioning is now
   `[X_z | Y_hat_z]` (history + forecast). If a cache has no `history` key it
   silently falls back to the old behaviour, so legacy runs still load.

## To launch
    # 1. regenerate residual caches (adds the history channel)
    for ds in pems08f pems08s pems04f pems04s; do
      for bb in mlp gcn gwn; do
        python scripts/extract_residuals.py --dataset $ds --backbone $bb --seed 42
      done
    done

    # 2. retrain both arms with --cond-past  (same grid as before)
    #    iso  -> --prior isotropic
    #    emp  -> --prior kronecker --temporal empirical

    # 3. evaluate under the locked protocol (best_crps ckpt, K=100, 50 steps, test)

    # 4. rebuild figures + tables
    python scripts/analysis/thesis_ch5_figures.py
    python scripts/analysis/paper_figures.py     # residual-structure figure only

## Thesis edits once the numbers land
- Remove the PROVISIONAL block + \fbox note at the top of §5.1.2.
- Update: Table 5.2 (all 12 cells x 3 metrics), Table 5.3 (both our rows),
  the paired-seed counts in §5.3 and the conclusion, the energy-score numbers
  in §5.4, and the coverage numbers in §5.4.
- Re-check: the "nine of twelve" / "34 of 42" claims are recomputed, not carried over.

# Normalized diffusion re-run — copy-paste runbook

6 cells: 3 datasets (`metrla`, `pemsbay`, `pems08f`) × 2 priors (`kronecker`, `isotropic`),
seed 0, **epochs 500 / patience 25**, normalized pipeline. Each eval depends on its
training job (`afterok`). GPUs: any working card (A40 or L40), Blackwell excluded.

---

## 1. Ship the code (run on your Mac)

```bash
cd /Users/serkan/Projects/research-tud
git add scripts/daic/clusters.sh scripts/daic/submit.sh \
        scripts/daic/submit_train_diffusion.sbatch scripts/daic/_common.sh \
        scripts/eval_diffusion.py scripts/train_diffusion.py \
        src/data/residual_dataset.py src/diffusion/prior.py
git commit -m "Normalized diffusion re-run: eval de-norm, any-gpu, 4h cap, log folders, lmod cache fix

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin structured-t-prior
```

(Already-committed files just won't show up — that's fine.)

---

## 2. Log in to the healthy node + pull (run on your Mac → lands you on daic02)

```bash
ssh daic-exp
```

Then **on daic02**:

```bash
cd ~/linuxhome/research-tud
git pull
```

---

## 3. One-time cleanup on daic02 (stale raw checkpoints + old flat logs)

```bash
rm -rf results/diffusion/{metrla,pemsbay,pems08f}/{kronecker,isotropic}/seed0
rm -f logs/diffusion/*.out logs/diffusion/*.err logs/eval/*.out logs/eval/*.err 2>/dev/null
```

---

## 4. Submit the 6 cells one-by-one (on daic02)

For each cell: run the **train** command, read the printed `Submitted batch job <TID>`,
put that number into the **eval** command's `--dependency=afterok:<TID>`.

The `set -x` trace shows the resource ask — confirm it reads
`--gres=gpu:1 --exclude=gpu[36-53]`.

### Cell 1 — metrla / kronecker
```bash
EPOCHS=500 PRIOR=kronecker EXTRA_FLAGS="--patience 25" \
  bash scripts/daic/submit.sh diffusion metrla 0
```
```bash
PRIOR=kronecker EVAL_ARGS="--dataset metrla --prior kronecker --seed 0 --k 30 --n-steps 50 --max-windows 2000 --gen-batch 1024 --w1-samples 4" \
  bash scripts/daic/submit.sh eval metrla 0 --dependency=afterok:<TID>
```

### Cell 2 — metrla / isotropic
```bash
EPOCHS=500 PRIOR=isotropic EXTRA_FLAGS="--patience 25" \
  bash scripts/daic/submit.sh diffusion metrla 0
```
```bash
PRIOR=isotropic EVAL_ARGS="--dataset metrla --prior isotropic --seed 0 --k 30 --n-steps 50 --max-windows 2000 --gen-batch 1024 --w1-samples 4" \
  bash scripts/daic/submit.sh eval metrla 0 --dependency=afterok:<TID>
```

### Cell 3 — pemsbay / kronecker
```bash
EPOCHS=500 PRIOR=kronecker EXTRA_FLAGS="--patience 25" \
  bash scripts/daic/submit.sh diffusion pemsbay 0
```
```bash
PRIOR=kronecker EVAL_ARGS="--dataset pemsbay --prior kronecker --seed 0 --k 30 --n-steps 50 --max-windows 2000 --gen-batch 1024 --w1-samples 4" \
  bash scripts/daic/submit.sh eval pemsbay 0 --dependency=afterok:<TID>
```

### Cell 4 — pemsbay / isotropic
```bash
EPOCHS=500 PRIOR=isotropic EXTRA_FLAGS="--patience 25" \
  bash scripts/daic/submit.sh diffusion pemsbay 0
```
```bash
PRIOR=isotropic EVAL_ARGS="--dataset pemsbay --prior isotropic --seed 0 --k 30 --n-steps 50 --max-windows 2000 --gen-batch 1024 --w1-samples 4" \
  bash scripts/daic/submit.sh eval pemsbay 0 --dependency=afterok:<TID>
```

### Cell 5 — pems08f / kronecker
```bash
EPOCHS=500 PRIOR=kronecker EXTRA_FLAGS="--patience 25" \
  bash scripts/daic/submit.sh diffusion pems08f 0
```
```bash
PRIOR=kronecker EVAL_ARGS="--dataset pems08f --prior kronecker --seed 0 --k 30 --n-steps 50 --max-windows 2000 --gen-batch 1024 --w1-samples 4" \
  bash scripts/daic/submit.sh eval pems08f 0 --dependency=afterok:<TID>
```

### Cell 6 — pems08f / isotropic
```bash
EPOCHS=500 PRIOR=isotropic EXTRA_FLAGS="--patience 25" \
  bash scripts/daic/submit.sh diffusion pems08f 0
```
```bash
PRIOR=isotropic EVAL_ARGS="--dataset pems08f --prior isotropic --seed 0 --k 30 --n-steps 50 --max-windows 2000 --gen-batch 1024 --w1-samples 4" \
  bash scripts/daic/submit.sh eval pems08f 0 --dependency=afterok:<TID>
```

---

## 5. Watch the queue (on daic02)

```bash
squeue -u $USER --format='%.10i %.28j %.2t %.11M %.19R'
```

Reason column: `(Resources)`/`(Priority)` = waiting for a GPU; `(Dependency)` = eval
waiting for its train; a node name (e.g. `gpu30`) = running. Should never be `gpu36`–`gpu53`.

Confirm a started job loaded modules + GPU cleanly:

```bash
ls -t logs/diffusion/*/*.out | head -1 | xargs tail -20
```

Look for the `─── job env ───` header with `python ... 3.11.9` and a GPU line.

---

## 6. When everything finishes — pull results back to the Mac

Run **on your Mac**:

```bash
rsync -avz --progress daic-exp:/tudelft.net/student-homes/a/sakin/research-tud-data/eval/ \
  ~/Projects/research-tud/results/eval/
```

Then tell me it's done and I'll rebuild the kronecker-vs-isotropic tables/figures and the
training-curve overview on the normalized checkpoints.
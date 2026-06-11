"""pick (n_steps, K) from an ablation json — the plateau knee.

prints two ints "N_STEPS K" to stdout. defensive: any missing/short/garbled
input falls back to sane defaults, so an eval job that reads this never breaks.

n_steps: smallest value whose Winkler, CRPS (and W1, looser) are within tolerance
         of the best in the sweep (more steps stop helping past it).
K:       smallest value whose coverage/CRPS match the high-K reference (more
         samples stop changing the estimate past it).
"""
import json
import sys

DEF_STEPS, DEF_K = 100, 30
STEPS_FLOOR, STEPS_CEIL = 50, 400
K_FLOOR, K_CEIL = 20, 100


def _knee_steps(ss):
    if len(ss) < 2:
        return None
    bw = min(r["winkler"] for r in ss)
    bc = min(r["crps"] for r in ss)
    bw1 = min(r["w1"] for r in ss)
    for r in sorted(ss, key=lambda x: x["n_steps"]):
        if r["winkler"] <= 1.02 * bw and r["crps"] <= 1.02 * bc and r["w1"] <= 1.05 * bw1:
            return r["n_steps"]
    return max(r["n_steps"] for r in ss)


def _knee_k(ks):
    if len(ks) < 2:
        return None
    ref = max(ks, key=lambda x: x["k"])          # most samples = most stable estimate
    for r in sorted(ks, key=lambda x: x["k"]):
        if abs(r["coverage"] - ref["coverage"]) <= 0.01 and 0.98 * ref["crps"] <= r["crps"] <= 1.02 * ref["crps"]:
            return r["k"]
    return ref["k"]


def pick(path):
    try:
        d = json.load(open(path))
        ns = _knee_steps(d.get("steps_sweep", [])) or DEF_STEPS
        k = _knee_k(d.get("k_sweep", [])) or DEF_K
    except Exception:
        ns, k = DEF_STEPS, DEF_K
    ns = max(STEPS_FLOOR, min(STEPS_CEIL, int(ns)))
    k = max(K_FLOOR, min(K_CEIL, int(k)))
    return ns, k


if __name__ == "__main__":
    ns, k = pick(sys.argv[1] if len(sys.argv) > 1 else "")
    print(ns, k)

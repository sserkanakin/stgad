"""update the reading companion html with the exact prior fit (prior_structures.json), the prior-alone
result (prior_alone.json) and the new chapter-5 figure/table numbering. edits the inlined D constant and a
few text blocks in place. cpu, mac.
usage: python scripts/analysis/companion_update.py <path to stgad_companion.html>
"""
import sys, json, re
H = sys.argv[1]; s = open(H).read()
struct = json.load(open("results/analysis/prior_structures.json")); alone = json.load(open("results/analysis/prior_alone.json"))
i = s.index("const D = ") + len("const D = "); j = s.index(";\n", i); D = json.loads(s[i:j])
for ds in D["datasets"]:
    P = struct[ds]["priors"]; ms = D["prior"][ds]["multiscale"]
    for k in ("iso", "road08", "matched", "cartesian"):
        ms[f"{k}_test"] = P[k]["ratio_test"]; ms[f"{k}_train"] = P[k]["ratio_train"]
    D["prior"][ds]["profile_test"] = P["matched"]["total_profile_test"]
    D["prior"][ds]["nll"] = {k: {"nll_test": P[k]["nll_test"]} for k in ("iso", "road08", "matched", "cartesian")}
D["alone"] = alone
s = s[:i] + json.dumps(D, separators=(",", ":")) + s[j:]

def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, old[:80]; s = s.replace(old, new, count)

# figure / table numbering of the rebuilt chapter 5
rep("Method: Section 4.2 · Figures 5.1 to 5.3", "Method: Section 4.2 · Figure 5.1")
rep("Method: Section 4.3 · Figures 5.4 to 5.6", "Method: Section 4.3, Appendix B · Figure 5.2, Table 5.2")
rep("Method: Section 4.5 · Table 5.3, Figure 5.7", "Method: Section 4.5 · Table 5.3")
rep("Method: Section 4.5 · Figures 5.8 to 5.10, Table 5.4", "Method: Section 4.5 · Figures 5.3 and 5.4, Tables 5.4 and 5.5")
rep("Method: Section 4.5 · Figures 5.11 and 5.12, Table 5.5", "Method: Section 4.5 · Figure 5.5, Table C.5")
rep("Method: Section 4.4.3 · Table 5.6, Figure 5.13", "Method: Section 4.4.3 · Table 5.6, Table C.6, Figure 5.6")

# R2: the exact fit, the product-graph prior, the lead-time caveat
rep("The prior fits at every scale; white noise collapses on sums", "The prior fits the sums to within about 15 %; white noise collapses on them")
rep("the matched prior is right at every scale, white noise is right per sensor and a factor 2 to 4 too narrow on the network total.",
    "the Kronecker prior is within about 15 % at every scale (0.83 to 1.09 on the network total), white noise is right per sensor and a factor 2 to 3 too narrow on the network total. Computed with the exact per-lead-time covariance; see the caveat below.")
rep('"matched" the prior used in the thesis.', '"Kronecker" the prior used in the thesis, "product graph" the product-graph GMRF of Chapter 2 matched by the same rule.')
rep("and the matched prior's held-out log-likelihood is <span id=\"r2nll\"></span> nats per entry better than white noise.",
    "and the Kronecker prior's held-out log-likelihood is <span id=\"r2nll\"></span> nats per entry better than white noise (the product-graph prior: <span id=\"r2nllc\"></span>).")
rep("<div class=\"read\"><b>Also reported:</b> the prior over-states neighbour correlation (matching one scale has a price), and a two-moment variant that matches both the total and the neighbours fits better in likelihood but was not retrained.</div>",
    "<div class=\"read\"><b>The caveat (Appendix B, Section 5.3):</b> the rule that picks γ read the prior's spread with an approximation (it ignored the mixing of the temporal modes), so the prior came out 7 to 37 % too wide on the training total; on the wider test windows it lands within 0.83 to 1.09. And a separable prior gives every lead time the same spatial correlation, while in the data the shared part grows along the horizon: the prior is 1.3 to 1.7 times too wide at 5 minutes ahead and 0.6 to 0.9 at one hour (<span id=\"r2prof\"></span> on the selected dataset). The product-graph prior has the same problem and fits worse in likelihood on every dataset.</div>")
rep("const lines = [['iso_test', 'white noise (iso)', P.muted, '4 3'], ['road08_test', 'road graph only, γ = 0.8', P.acc, '2 3'], ['matched_test', 'matched prior (used)', P.warm, '']];",
    "const lines = [['iso_test', 'white noise (iso)', P.muted, '4 3'], ['road08_test', 'road graph only, γ = 0.8', P.acc, '2 3'], ['cartesian_test', 'product-graph prior', P.violet || '#6A4FBF', '6 3'], ['matched_test', 'Kronecker prior (used)', P.warm, '']];")
rep("document.getElementById('r2nll').textContent = (D.prior[d].nll.iso.nll_test - D.prior[d].nll.matched.nll_test).toFixed(2); }",
    "document.getElementById('r2nll').textContent = (D.prior[d].nll.iso.nll_test - D.prior[d].nll.matched.nll_test).toFixed(2); document.getElementById('r2nllc').textContent = (D.prior[d].nll.iso.nll_test - D.prior[d].nll.cartesian.nll_test).toFixed(2); const pf = D.prior[d].profile_test; document.getElementById('r2prof').textContent = pf[0].toFixed(2) + ' at 5 min, ' + pf[11].toFixed(2) + ' at one hour'; }")

# R3: the prior alone as a column
rep("return {d, iso: c.iso.crps, kron: c.kron.crps, best}; });",
    "return {d, iso: c.iso.crps, kron: c.kron.crps, alone: D.alone[d].kron.crps, best}; });")
rep("<th>best published per-node CRPS</th><th>white noise</th><th>Kronecker prior</th><th>Kronecker vs white</th></tr></thead>",
    "<th>best published per-node CRPS</th><th>prior alone (no denoiser)</th><th>white noise</th><th>Kronecker prior</th><th>Kronecker vs white</th></tr></thead>")
rep("</td><td>${r.iso.toFixed(4)}</td><td>${r.kron.toFixed(4)}</td>", "</td><td>${r.alone.toFixed(4)}</td><td>${r.iso.toFixed(4)}</td><td>${r.kron.toFixed(4)}</td>")
rep("Per node: nothing lost, and both below the published methods", "Per node: nothing lost, both below the published methods, and the prior alone already there")
rep("The advantage over the published methods comes from the backbone and the residual design, not from the prior.",
    "The advantage over the published methods comes from the backbone and the residual design, which the prior alone already has: the Gaussian residual model with no denoiser is 6 to 8 % below the best published number, and the trained model is sharper than it by a few percent per node.")

# R4: the prior alone on the total
rep("Both models are still under-dispersed on the total; the prior closes part of the gap.</div></div>",
    "Both trained models are still under-dispersed on the total; the prior closes part of the gap. The prior used on its own is nearly calibrated on the total (coverage <span id=\"r4alone\"></span>), so the denoiser is what narrows it; Result 6 says which pattern.</div></div>")
rep("(function () { const P = PAL(); const g = D.cells.gwn; barChart(document.getElementById('r4svg')",
    "(function () { const a = D.datasets.map(d => D.alone[d].kron.joint_tot_coverage); document.getElementById('r4alone').textContent = Math.min(...a).toFixed(2) + ' to ' + Math.max(...a).toFixed(2) + ' against 0.55 to 0.74 for the trained models'; })();\n(function () { const P = PAL(); const g = D.cells.gwn; barChart(document.getElementById('r4svg')")
open(H, "w").write(s); print("companion updated")

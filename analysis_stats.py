"""Reviewer-requested re-analyses computed from the per-pair records.

Reads results.json (full-resolution, Table 1) and compare_proc512.json
(matched 512 px, Table 2). Prints everything needed for the revised tables.
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

ROOT = sys.argv[1]
rng = np.random.default_rng(0)

def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])

def paired(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    lo, hi = boot_ci(d)
    wins = int((d > 0).sum()); ties = int((d == 0).sum())
    print(f"  {label}: mean diff {d.mean():+.4f} [95% CI {lo:+.4f},{hi:+.4f}] "
          f"wins {wins}/{len(d)} (ties {ties}) p={p:.3g}")
    return d.mean(), lo, hi, p

# ------------------------------------------------------------------ Table 1
print("=" * 70, "\nFULL-RESOLUTION (results.json)")
R = json.load(open(f"{ROOT}/results.json"))
idx = defaultdict(dict)
for r in R["records"]:
    idx[(r["scenario"], r["pair"])][r["method"]] = r
keys = sorted(idx)
sift_ok = [k for k in keys if idx[k]["P1_fidelity"]["aligned"]]
sift_fail = [k for k in keys if not idx[k]["P1_fidelity"]["aligned"]]
print("SIFT-aligned pairs:", len(sift_ok), " failed:", len(sift_fail))
print("SIFT inlier counts on aligned pairs:",
      sorted(idx[k]["P1_fidelity"]["n_inliers"] for k in sift_ok))
print("LightGlue inlier counts (P3):",
      sorted(idx[k]["P3_fidelity_lg"]["n_inliers"] for k in keys))
print("B2 homography inliers:", sorted(idx[k]["B2_homog_alpha"]["n_inliers"] for k in keys))
per_scen_fail = defaultdict(int)
for k in sift_fail: per_scen_fail[k[0]] += 1
print("SIFT failures per scenario:", dict(per_scen_fail))

def agg(method, subset, metrics=("ssim", "psnr", "lpips", "niqe", "lapvar")):
    out = {}
    for m in metrics:
        v = np.array([idx[k][method][m] for k in subset], float)
        out[m] = (v.mean(), *boot_ci(v)) if m != "lapvar" else (np.median(v), *np.percentile(v, [25, 75]))
    return out

def row(label, method, subset):
    a = agg(method, subset)
    print(f"{label:34s} n={len(subset):2d} SSIM {a['ssim'][0]:.3f} [{a['ssim'][1]:.3f},{a['ssim'][2]:.3f}] "
          f"PSNR {a['psnr'][0]:.2f} LPIPS {a['lpips'][0]:.3f} NIQE {a['niqe'][0]:.2f} LapVar med {a['lapvar'][0]:.1f}")

print("\n-- (a) common SIFT-aligned subset (34 pairs) --")
for lab, m in [("Input", "Base_M1"), ("Single-image Lanczos (B1)", "B1_lanczos"),
               ("Homography+alpha-blend (B2)", "B2_homog_alpha"),
               ("Similarity+alpha-blend (B3)", "B3_affine_alpha"),
               ("SIFT, fusion + aggressive post (V5)", "V5_full"),
               ("SIFT, fusion + light post (P2)", "P2_light_post"),
               ("SIFT, fusion, no post (P1)", "P1_fidelity"),
               ("LightGlue, fusion + aggr. post (A1)", "A1_lightglue"),
               ("LightGlue, fusion, no post (P3=prop)", "P3_fidelity_lg")]:
    row(lab, m, sift_ok)
print("\n-- (b) full set, 55 pairs, fallback (resize) included where registration failed --")
for lab, m in [("Input", "Base_M1"), ("Single-image Lanczos (B1)", "B1_lanczos"),
               ("Homography+alpha-blend (B2)", "B2_homog_alpha"),
               ("SIFT, fusion + aggressive post (V5)", "V5_full"),
               ("SIFT, fusion, no post (P1)", "P1_fidelity"),
               ("LightGlue, fusion + aggr. post (A1)", "A1_lightglue"),
               ("LightGlue, fusion, no post (P3=prop)", "P3_fidelity_lg")]:
    row(lab, m, keys)
print("\n-- (c) the 21 SIFT-failure pairs only --")
for lab, m in [("Input", "Base_M1"), ("P1 (fallback=resize of M1)", "P1_fidelity"), ("P3 LightGlue", "P3_fidelity_lg")]:
    row(lab, m, sift_fail)

print("\n-- paired tests, full res --")
for met in ["ssim", "lpips", "niqe"]:
    print(met)
    paired([idx[k]["P1_fidelity"][met] for k in sift_ok], [idx[k]["V5_full"][met] for k in sift_ok], "P1 (no post) - V5 (post), 34 common")
    paired([idx[k]["P3_fidelity_lg"][met] for k in sift_ok], [idx[k]["P1_fidelity"][met] for k in sift_ok], "P3 (LG) - P1 (SIFT), 34 common")
    paired([idx[k]["P3_fidelity_lg"][met] for k in keys], [idx[k]["Base_M1"][met] for k in keys], "P3 - input, 55")
    paired([idx[k]["P3_fidelity_lg"][met] for k in keys], [idx[k]["B1_lanczos"][met] for k in keys], "P3 - single-image Lanczos, 55")
    paired([idx[k]["P3_fidelity_lg"][met] for k in sift_fail], [idx[k]["Base_M1"][met] for k in sift_fail], "P3 - input, 21 SIFT-fail pairs")

print("\n-- per-scenario, P3 vs input, full res --")
for s in ["S1", "S2", "S3", "S4", "S5"]:
    ks = [k for k in keys if k[0] == s]
    for met in ["ssim", "psnr", "lpips", "niqe"]:
        a = np.mean([idx[k]["P3_fidelity_lg"][met] for k in ks]); b = np.mean([idx[k]["Base_M1"][met] for k in ks])
        print(f"  {s} {met}: prop {a:.3f} input {b:.3f} rel {100*(a-b)/b:+.1f}%")

# ------------------------------------------------------------------ Table 2
print("\n" + "=" * 70, "\nMATCHED 512 px (compare_proc512.json)")
C = json.load(open(f"{ROOT}/compare_proc512.json"))
cidx = defaultdict(dict)
for r in C["records"]:
    cidx[(r["scenario"], r["pair"])][r["method"]] = r
ckeys = sorted(cidx)
meths = ["Input", "Proposed (single)", "Proposed", "TTSR-rec"]
ocr_keys = [k for k in ckeys if all("char_acc" in cidx[k][m] for m in meths)]
print("pairs:", len(ckeys), " pairs with OCR:", len(ocr_keys))
print("aligned (Proposed single):", sum(cidx[k]["Proposed (single)"]["aligned"] for k in ckeys))
for m in meths:
    for met in ["ssim", "psnr", "lpips", "niqe"]:
        v = np.array([cidx[k][m][met] for k in ckeys])
        lo, hi = boot_ci(v)
        print(f"  {m:18s} {met:5s} {v.mean():.3f} [{lo:.3f},{hi:.3f}]")
    for met in ["char_acc", "word_acc"]:
        v = np.array([cidx[k][m][met] for k in ocr_keys])
        lo, hi = boot_ci(v)
        print(f"  {m:18s} {met:8s} {v.mean():.3f} [{lo:.3f},{hi:.3f}]  median {np.median(v):.3f}")
    print(f"  {m:18s} runtime_ms mean {np.mean([cidx[k][m]['runtime_ms'] for k in ckeys]):.0f}")

print("\n-- paired OCR tests (pair level) --")
for met in ["char_acc", "word_acc"]:
    print(met)
    g = lambda m: [cidx[k][m][met] for k in ocr_keys]
    paired(g("Proposed (single)"), g("Input"), "Proposed - Input")
    paired(g("TTSR-rec"), g("Input"), "TTSR - Input")
    paired(g("TTSR-rec"), g("Proposed (single)"), "TTSR - Proposed")
    paired(g("Proposed"), g("Input"), "Proposed(MS) - Input")
print("\n-- paired fidelity tests (pair level, 55) --")
for met in ["ssim", "psnr", "lpips", "niqe"]:
    print(met)
    g = lambda m: [cidx[k][m][met] for k in ckeys]
    paired(g("Proposed (single)"), g("Input"), "Proposed - Input")
    paired(g("TTSR-rec"), g("Input"), "TTSR - Input")
    paired(g("TTSR-rec"), g("Proposed (single)"), "TTSR - Proposed")

print("\n-- scene level (n=11), OCR char acc --")
scenes = sorted(set(k[1] for k in ocr_keys))
def scene_means(m, met="char_acc"):
    return [np.mean([cidx[k][m][met] for k in ocr_keys if k[1] == s]) for s in scenes]
for met in ["char_acc", "word_acc"]:
    print(met)
    paired(scene_means("Proposed (single)", met), scene_means("Input", met), "Proposed - Input (scene)")
    paired(scene_means("TTSR-rec", met), scene_means("Input", met), "TTSR - Input (scene)")
    paired(scene_means("TTSR-rec", met), scene_means("Proposed (single)", met), "TTSR - Proposed (scene)")

print("\n-- per-scenario OCR (char / word), matched 512 --")
for s in ["S1", "S2", "S3", "S4", "S5"]:
    ks = [k for k in ocr_keys if k[0] == s]
    line = f"  {s} (n={len(ks)}): "
    for m in ["Input", "Proposed (single)", "TTSR-rec"]:
        c = np.mean([cidx[k][m]["char_acc"] for k in ks]); w = np.mean([cidx[k][m]["word_acc"] for k in ks])
        line += f"{m[:9]} c={c:.3f} w={w:.3f} | "
    # wins vs input
    pw = sum(cidx[k]["Proposed (single)"]["char_acc"] > cidx[k]["Input"]["char_acc"] for k in ks)
    tw = sum(cidx[k]["TTSR-rec"]["char_acc"] > cidx[k]["Input"]["char_acc"] for k in ks)
    print(line + f"wins vs input: prop {pw}/{len(ks)} ttsr {tw}/{len(ks)}")
print("\n-- per-scenario SSIM/NIQE, matched 512 --")
for s in ["S1", "S2", "S3", "S4", "S5"]:
    ks = [k for k in ckeys if k[0] == s]
    line = f"  {s}: "
    for m in ["Input", "Proposed (single)", "TTSR-rec"]:
        line += f"{m[:9]} ssim={np.mean([cidx[k][m]['ssim'] for k in ks]):.3f} niqe={np.mean([cidx[k][m]['niqe'] for k in ks]):.2f} lpips={np.mean([cidx[k][m]['lpips'] for k in ks]):.3f} | "
    print(line)
# Input char-acc distribution
v = np.array([cidx[k]["Input"]["char_acc"] for k in ocr_keys])
print("\nInput char-acc: zeros", int((v == 0).sum()), "min", v.min(), "max", v.max())
v = np.array([cidx[k]["TTSR-rec"]["char_acc"] for k in ocr_keys])
print("TTSR char-acc: zeros", int((v == 0).sum()), "min", v.min(), "max", v.max(), "std", v.std())
# NIQE reduction matched
a = np.mean([cidx[k]["Proposed (single)"]["niqe"] for k in ckeys]); b = np.mean([cidx[k]["Input"]["niqe"] for k in ckeys])
print(f"NIQE matched: {b:.2f}->{a:.2f} ({100*(a-b)/b:+.1f}%)")
a = np.mean([idx[k]["P3_fidelity_lg"]["niqe"] for k in keys]); b = np.mean([idx[k]["Base_M1"]["niqe"] for k in keys])
print(f"NIQE full: {b:.2f}->{a:.2f} ({100*(a-b)/b:+.1f}%)")
# LPIPS win rate full res
w = sum(idx[k]["P3_fidelity_lg"]["lpips"] < idx[k]["Base_M1"]["lpips"] for k in keys)
print("LPIPS wins P3 vs input full res:", w, "/55;  SSIM wins:", sum(idx[k]["P3_fidelity_lg"]["ssim"] > idx[k]["Base_M1"]["ssim"] for k in keys))
w = sum(cidx[k]["Proposed (single)"]["lpips"] < cidx[k]["Input"]["lpips"] for k in ckeys)
print("LPIPS wins matched:", w, "/55; NIQE wins:", sum(cidx[k]["Proposed (single)"]["niqe"] < cidx[k]["Input"]["niqe"] for k in ckeys))

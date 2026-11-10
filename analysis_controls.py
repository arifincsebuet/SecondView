import json, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
rng = np.random.default_rng(0)
D = sys.argv[1]
recs = [json.loads(l) for l in open(f"{D}/controls_512.jsonl")] + [json.loads(l) for l in open(f"{D}/fusion_512.jsonl")]
idx = defaultdict(dict)
for r in recs: idx[(r["scenario"], r["pair"])][r["method"]] = r
keys = sorted(idx)
fus_ok = [k for k in keys if idx[k]["Fusion"]["aligned"]]
print("fusion aligned", len(fus_ok), "matchers", set(idx[k]["Fusion"]["matcher"] for k in keys))

def ci(x):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])
def paired(a, b, lab):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    try: p = wilcoxon(a, b).pvalue
    except ValueError: p = np.nan
    lo, hi = ci(d)
    print(f"   {lab}: diff {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] a>b {int((d>0).sum())}/{len(d)} ties {int((d==0).sum())} p={p:.3g}")

meths = ["Input", "DownUp-4x@384", "DownUp-4x@full", "Unsharp-0.5", "Unsharp-1.2", "Fusion"]
for subset, name in [(keys, "ALL 55"), (fus_ok, f"FUSION-ALIGNED {len(fus_ok)}")]:
    print("\n=== ", name)
    for m in meths + ["Chance"]:
        line = f"{m:15s}"
        for met in ["ssim", "psnr", "lapvar", "char_acc", "word_acc"]:
            v = [idx[k][m][met] for k in subset if met in idx[k][m]]
            if v:
                lo, hi = ci(v)
                line += f" {met} {np.mean(v):.3f} [{lo:.3f},{hi:.3f}]" if met != "lapvar" else f" lapvar med {np.median(v):.1f}"
        print(line)
    for met in ["ssim", "psnr", "char_acc", "word_acc"]:
        print(" ", met)
        g = lambda m: [idx[k][m][met] for k in subset]
        paired(g("Unsharp-1.2"), g("Input"), "Unsharp1.2 - Input")
        paired(g("Unsharp-0.5"), g("Input"), "Unsharp0.5 - Input")
        paired(g("Fusion"), g("Input"), "Fusion - Input")
        paired(g("Fusion"), g("Unsharp-0.5"), "Fusion - Unsharp0.5")
        paired(g("Fusion"), g("Unsharp-1.2"), "Fusion - Unsharp1.2")
        paired(g("DownUp-4x@384"), g("Input"), "DownUp384 - Input")
        if met == "char_acc":
            paired(g("Input"), g("Chance"), "Input - Chance")
            paired(g("Unsharp-1.2"), g("Chance"), "Unsharp1.2 - Chance")
            paired(g("Fusion"), g("Chance"), "Fusion - Chance")

print("\n=== per scenario (all 55) char acc")
for s in ["S1","S2","S3","S4","S5"]:
    ks = [k for k in keys if k[0]==s]
    print(s, " ".join(f"{m[:10]}={np.mean([idx[k][m]['char_acc'] for k in ks]):.3f}" for m in meths+["Chance"]))
print("=== per scenario word acc")
for s in ["S1","S2","S3","S4","S5"]:
    ks = [k for k in keys if k[0]==s]
    print(s, " ".join(f"{m[:10]}={np.mean([idx[k][m]['word_acc'] for k in ks]):.3f}" for m in meths))
print("ref lengths:", sorted(idx[k]["Chance"]["ref_len"] for k in keys))
# chance vs TTSR from compare file
C = json.load(open(sys.argv[2]))
cidx = defaultdict(dict)
for r in C["records"]: cidx[(r["scenario"], r["pair"])][r["method"]] = r
paired([cidx[k]["TTSR-rec"]["char_acc"] for k in keys], [idx[k]["Chance"]["char_acc"] for k in keys], "TTSR(Table2) - Chance")
paired([cidx[k]["Input"]["char_acc"] for k in keys], [idx[k]["Chance"]["char_acc"] for k in keys], "Input(Table2) - Chance")
paired([cidx[k]["Proposed (single)"]["char_acc"] for k in keys], [idx[k]["Chance"]["char_acc"] for k in keys], "Proposed(Table2) - Chance")
# Pairs where Table-2 input char acc > chance
v = [cidx[k]["Input"]["char_acc"] > idx[k]["Chance"]["char_acc"] for k in keys]
print("Table2 Input above chance on", sum(v), "/55; TTSR above chance on", sum(cidx[k]["TTSR-rec"]["char_acc"] > idx[k]["Chance"]["char_acc"] for k in keys))
# Scenario where input is above chance
for s in ["S1","S2","S3","S4","S5"]:
    ks=[k for k in keys if k[0]==s]
    print(s, "input>chance", sum(cidx[k]["Input"]["char_acc"] > idx[k]["Chance"]["char_acc"] for k in ks), "ttsr>chance", sum(cidx[k]["TTSR-rec"]["char_acc"] > idx[k]["Chance"]["char_acc"] for k in ks), "chance mean", round(np.mean([idx[k]["Chance"]["char_acc"] for k in ks]),3))

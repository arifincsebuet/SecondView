"""
cvpr_figures.py -- vector (PDF) figures for the CVPR revision
==============================================================
Reads results.json, compare_proc512.json, controls_512.jsonl, fusion_512.jsonl
and runtime_colab.json; writes PDFs into ../Pics/.

  fig_alignment.pdf     registration success per scenario, SIFT vs LightGlue
  fig_tradeoff.pdf      SSIM vs edge contrast for the classical variants (34 common pairs)
  fig_ocr_scenario.pdf  per-scenario character / word accuracy with chance level
  fig_controls.pdf      paired differences vs input for the single-image / protocol controls
  fig_runtime.pdf       per-stage latency
  fig_qualitative.png   crops (raster photos)

Usage: python cvpr_figures.py [--pics ../Pics]
"""
import json, sys, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_IN, C_PROP, C_TTSR, C_UNS, C_OVER, C_GREY = "#4C72B0", "#2a9d5c", "#C44E52", "#DD8452", "#8172B2", "#777777"
SCEN = ["S1", "S2", "S3", "S4", "S5"]
SCEN_LAB = ["D1\nmild blur", "D2\nstrong blur", "D3\nmotion", "D4\nmotion+JPEG", "D5\nlow light"]
rng = np.random.default_rng(0)


def ci(x, n=10000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def load_idx(records):
    idx = defaultdict(dict)
    for r in records:
        idx[(r["scenario"], r["pair"])][r["method"]] = r
    return idx


def fig_alignment(idx, out):
    sift = [100 * np.mean([idx[k]["P1_fidelity"]["aligned"] for k in idx if k[0] == s]) for s in SCEN]
    lg = [100 * np.mean([idx[k]["P3_fidelity_lg"]["aligned"] for k in idx if k[0] == s]) for s in SCEN]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.bar(x - w / 2, sift, w, color=C_OVER, label="SIFT + RANSAC", zorder=3)
    ax.bar(x + w / 2, lg, w, color=C_PROP, label="SuperPoint + LightGlue + RANSAC", zorder=3)
    for i, (a, b) in enumerate(zip(sift, lg)):
        ax.text(i - w / 2, a + 2, f"{a:.0f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, b + 2, f"{b:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(SCEN_LAB, fontsize=6.5); ax.set_ylim(0, 108)
    ax.set_ylabel("Registered pairs (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, framealpha=0.9, fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3); fig.savefig(out / "fig_alignment.pdf"); plt.close(fig)


def fig_tradeoff(idx, out):
    common = [k for k in idx if idx[k]["P1_fidelity"]["aligned"]]
    rows = [("Input", "Base_M1", C_IN, "o"), ("Lanczos (single image)", "B1_lanczos", C_GREY, "D"),
            ("Homography + alpha blend", "B2_homog_alpha", "#937860", "^"),
            ("Similarity + alpha blend", "B3_affine_alpha", "#937860", "v"),
            ("Fusion + aggressive post", "V5_full", C_OVER, "s"), ("Fusion + light post", "P2_light_post", C_UNS, "P"),
            ("Fusion, no post (SIFT)", "P1_fidelity", "#1f6f45", "X"), ("Fusion, no post (LightGlue)", "P3_fidelity_lg", C_PROP, "*")]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for lab, m, c, mk in rows:
        s = np.mean([idx[k][m]["ssim"] for k in common]); l = np.median([idx[k][m]["lapvar"] for k in common])
        ax.scatter(l, s, color=c, marker=mk, s=55 if mk == "*" else 32, label=lab, zorder=3, edgecolor="k", linewidth=0.4)
    ax.set_xscale("log"); ax.set_xlabel("Edge contrast (median Laplacian variance, log)")
    ax.set_ylabel("SSIM vs. ground truth"); ax.set_ylim(0.62, 0.86)
    ax.legend(fontsize=5.6, loc="lower left", framealpha=0.9, ncol=1)
    fig.tight_layout(pad=0.3); fig.savefig(out / "fig_tradeoff.pdf"); plt.close(fig)


def fig_ocr_scenario(cidx, ctrl, out):
    keys = sorted(cidx)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    meths = [("Input", "Input", C_IN), ("Proposed (single)", "Two-view fusion", C_PROP), ("TTSR-rec", "TTSR-rec", C_TTSR)]
    x = np.arange(5); w = 0.26
    for ax, met, ylab in zip(axes, ["char_acc", "word_acc"], ["Character accuracy (1$-$CER)", "Word accuracy (1$-$WER)"]):
        for j, (m, lab, c) in enumerate(meths):
            means, lo, hi = [], [], []
            for s in SCEN:
                v = [cidx[k][m][met] for k in keys if k[0] == s]
                means.append(np.mean(v)); a, b = ci(v); lo.append(means[-1] - a); hi.append(b - means[-1])
            ax.bar(x + (j - 1) * w, means, w, yerr=[lo, hi], color=c, label=lab, zorder=3, error_kw=dict(lw=0.6, capsize=1.5))
        if met == "char_acc":
            ch = [np.mean([ctrl[k]["Chance"]["char_acc"] for k in keys if k[0] == s]) for s in SCEN]
            ax.plot(x, ch, "k--", lw=0.9, label="Chance (length-matched random text)", zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(SCEN_LAB, fontsize=6.5); ax.set_ylabel(ylab)
        ax.legend(fontsize=5.8, framealpha=0.9, loc="upper right" if met == "char_acc" else "upper center")
    axes[0].set_ylim(0, 0.36); axes[1].set_ylim(0, 0.2)
    fig.tight_layout(pad=0.3, w_pad=1.0); fig.savefig(out / "fig_ocr_scenario.pdf"); plt.close(fig)


def fig_controls(ctrl, out):
    keys = sorted(ctrl)
    meths = [("DownUp-4x@384", "DU@384"), ("DownUp-4x@full", "DU@full"),
             ("Unsharp-0.5", "US 0.5"), ("Unsharp-1.2", "US 1.2"), ("Fusion", "Fusion")]
    cols = [C_TTSR, "#e8a0a0", C_UNS, "#b8641e", C_PROP]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, met, lab, lim in zip(axes, ["ssim", "char_acc", "word_acc"],
                                 ["$\\Delta$SSIM vs. input", "$\\Delta$ char. acc. vs. input", "$\\Delta$ word acc. vs. input"],
                                 [(-0.11, 0.02), (-0.13, 0.05), (-0.04, 0.04)]):
        for i, ((m, ml), c) in enumerate(zip(meths, cols)):
            d = np.array([ctrl[k][m][met] - ctrl[k]["Input"][met] for k in keys])
            a, b = ci(d)
            ax.bar(i, d.mean(), color=c, zorder=3, width=0.65)
            ax.errorbar(i, d.mean(), yerr=[[d.mean() - a], [b - d.mean()]], color="k", lw=0.7, capsize=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(meths))); ax.set_xticklabels([ml for _, ml in meths], fontsize=6.5, rotation=30, ha="right")
        ax.set_ylabel(lab); ax.set_ylim(*lim)
    fig.tight_layout(pad=0.3, w_pad=1.0); fig.savefig(out / "fig_controls.pdf"); plt.close(fig)


def fig_runtime(bench, out):
    st = bench["stages"]
    labels = ["S1 registration", "S2 dense flow", "S3–S4 weighting + fusion", "S5 post-processing\n(not in recommended config.)"]
    vals = [st["s2_registration_ms"]["mean"], st["s3_flow_ms"]["mean"], st["s4s5_fusion_ms"]["mean"], st["s5_post_ms"]["mean"]]
    errs = [st["s2_registration_ms"]["std"], st["s3_flow_ms"]["std"], st["s4s5_fusion_ms"]["std"], st["s5_post_ms"]["std"]]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    cols = [C_IN, C_IN, C_PROP, C_GREY]
    bars = ax.barh(range(4), vals, xerr=errs, color=cols, zorder=3, error_kw=dict(lw=0.6, capsize=2))
    bars[3].set_hatch("///"); bars[3].set_alpha(0.6)
    for i, v in enumerate(vals):
        ax.text(v + 60, i, f"{v:.0f} ms", va="center", fontsize=6.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel(f"Latency per {bench['image_size']} frame (ms; mean $\\pm$ s.d., {bench['repeats']} runs)", fontsize=7)
    ax.set_xlim(0, 1400)
    fig.tight_layout(pad=0.3); fig.savefig(out / "fig_runtime.pdf"); plt.close(fig)


def fig_qualitative(ds, ttsr_dir, out, pairs=(("S1", "pair_05"), ("S4", "pair_02")), box=None):
    """Crops: input / unsharp / two-view fusion / TTSR / ground truth."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import enhance_core as ec
    from controls_eval import unsharp
    rows = []
    for s, p in pairs:
        d = ds / s / p
        m1, m2, gt = (cv2.imread(str(d / f)) for f in ("m1.jpg", "m2.jpg", "gt.jpg"))
        r = ec.enhance(m1, m2, matcher="auto", multiscale=False, guided=False, use_post=False)
        fus = r["result"]
        rows.append((f"{s.replace('S','D')} / {p}  ({r['matcher_used']}, {r['n_inliers']} RANSAC inliers)", [m1, unsharp(m1, 1.2), fus, gt], r["roi"]))
    titles = ["Degraded input", "Single-image unsharp ($\\alpha$=1.2)", "Two-view fusion", "Ground truth"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.75 * len(rows)))
    for i, (lab, imgs, roi) in enumerate(rows):
        x1, y1, x2, y2 = roi if roi else (0, 0, imgs[0].shape[1], imgs[0].shape[0])
        # crop a text-rich window in the middle of the ROI
        cw, chh = int(0.40 * (x2 - x1)), int(0.20 * (y2 - y1))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bx = (cx - cw // 2, cy - chh // 2, cx + cw // 2, cy + chh // 2)
        for j, im in enumerate(imgs):
            crop = im[bx[1]:bx[3], bx[0]:bx[2]]
            axes[i, j].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); axes[i, j].axis("off"); axes[i, j].grid(False)
            if i == 0: axes[i, j].set_title(titles[j], fontsize=7)
        axes[i, 0].text(0.0, -0.08, lab, transform=axes[i, 0].transAxes, fontsize=6, va="top")
    fig.tight_layout(pad=0.2, h_pad=0.9); fig.savefig(out / "fig_qualitative.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pics", default="../Pics"); ap.add_argument("--controls", default=".")
    ap.add_argument("--dataset", default="dataset"); ap.add_argument("--ttsr", default="refsr_sr_out")
    ap.add_argument("--skip-qual", action="store_true")
    a = ap.parse_args()
    out = Path(a.pics); out.mkdir(exist_ok=True)
    idx = load_idx(json.load(open("results.json"))["records"])
    cidx = load_idx(json.load(open("compare_proc512.json"))["records"])
    ctrl_recs = [json.loads(l) for f in ("controls_512.jsonl", "fusion_512.jsonl") for l in open(Path(a.controls) / f)]
    ctrl = load_idx(ctrl_recs)
    fig_alignment(idx, out); print("alignment")
    fig_tradeoff(idx, out); print("tradeoff")
    fig_ocr_scenario(cidx, ctrl, out); print("ocr")
    fig_controls(ctrl, out); print("controls")
    fig_runtime(json.load(open("runtime_colab.json")), out); print("runtime")
    if not a.skip_qual:
        fig_qualitative(Path(a.dataset), Path(a.ttsr), out); print("qualitative")


if __name__ == "__main__":
    main()

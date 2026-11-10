"""
cvpr_figures_v2.py -- redrawn vector figures for the CVPR submission.

Changes from cvpr_figures.py:
  * uncertainty shown as point + interval (no bars with whiskers), so value
    labels, intervals and legends never overlap and nothing is clipped;
  * axis limits computed from the data and intervals;
  * legends placed outside the plotting area; direct labels where possible;
  * figures drawn at their final printed width (column 3.25 in, text 6.875 in)
    in a Times-like font, so text appears at true size in the paper;
  * one validated categorical palette (blue / orange / aqua), with distinct
    markers as a second cue for grayscale printing and color-vision deficiency.

Reads results.json, compare_proc512.json, controls_512.json, fusion_512.json,
runtime_colab.json. Writes *_v2.pdf into --pics.
"""
import json, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
          "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf",
          "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf"]:
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLW, TEXTW = 3.25, 6.875

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
SCEN = ["S1", "S2", "S3", "S4", "S5"]
DEG = ["D1", "D2", "D3", "D4", "D5"]
DEG_LONG = ["D1\nblur 2.5", "D2\nblur 4.5", "D3\nmotion", "D4\nmotion+JPEG", "D5\nlow light"]
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
    sift = [sum(idx[k]["P1_fidelity"]["aligned"] for k in idx if k[0] == s) for s in SCEN]
    lg = [sum(idx[k]["P3_fidelity_lg"]["aligned"] for k in idx if k[0] == s) for s in SCEN]
    x = np.arange(5); w = 0.36
    fig, ax = plt.subplots(figsize=(COLW, 1.55))
    ax.bar(x - w / 2 - 0.01, sift, w, color=BLUE, label="SIFT + RANSAC", zorder=3)
    ax.bar(x + w / 2 + 0.01, lg, w, color=ORANGE, label="SuperPoint + LightGlue + RANSAC", zorder=3,
           hatch="////", edgecolor="white", linewidth=0)
    for i, (a, b) in enumerate(zip(sift, lg)):
        ax.text(i - w / 2 - 0.01, a + 0.25, f"{a}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.text(i + w / 2 + 0.01, b + 0.25, f"{b}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(DEG_LONG, fontsize=6.5)
    ax.set_ylim(0, 12.8); ax.set_yticks([0, 3, 6, 9, 11])
    ax.set_ylabel("Registered pairs (of 11)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)
    fig.savefig(out / "fig_alignment_v2.pdf"); plt.close(fig)


def fig_tradeoff(idx, out):
    common = [k for k in idx if idx[k]["P1_fidelity"]["aligned"]]
    # (label, method, color, marker, text offset in points, ha)
    rows = [("Input", "Base_M1", INK2, "o", (-4, -9), "center"),
            ("Lanczos", "B1_lanczos", INK2, "D", (4, 4), "left"),
            ("Homography + blend", "B2_homog_alpha", BLUE, "^", (-5, -2), "right"),
            ("Similarity + blend", "B3_affine_alpha", BLUE, "v", (5, -2), "left"),
            ("Fusion + aggressive post", "V5_full", ORANGE, "s", (-6, 2), "right"),
            ("Fusion + light post", "P2_light_post", ORANGE, "P", (-6, 2), "right"),
            ("Fusion, no post (SIFT, LightGlue)", "P1_fidelity", AQUA, "X", (6, -8), "left"),
            (None, "P3_fidelity_lg", AQUA, "*", None, None)]
    fig, ax = plt.subplots(figsize=(COLW, 2.05))
    for lab, m, c, mk, off, ha in rows:
        s = np.mean([idx[k][m]["ssim"] for k in common]); l = np.median([idx[k][m]["lapvar"] for k in common])
        ax.scatter(l, s, color=c, marker=mk, s=42 if mk == "*" else 24, zorder=3, edgecolor="white", linewidth=0.6)
        if lab:
            ax.annotate(lab, (l, s), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=6.5, color=INK)
    ax.set_xscale("log"); ax.set_xlim(2, 900)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(FixedFormatter(["3", "10", "30", "100", "300"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Edge contrast (median Laplacian variance, log scale)")
    ax.set_ylabel("SSIM vs. ground truth"); ax.set_ylim(0.645, 0.85)
    fig.savefig(out / "fig_tradeoff_v2.pdf"); plt.close(fig)


def _point_interval(ax, x, m, lo, hi, color, marker, label=None, ms=4.2):
    ax.vlines(x, lo, hi, color=color, lw=1.1, zorder=3)
    ax.plot(x, m, linestyle="none", marker=marker, ms=ms, color=color, markeredgecolor="white",
            markeredgewidth=0.5, zorder=4, label=label)


def fig_ocr_scenario(cidx, ctrl, out):
    keys = sorted(cidx)
    meths = [("Input", "Input", BLUE, "o"), ("Proposed (single)", "Two-view fusion", ORANGE, "s"),
             ("TTSR-rec", "TTSR-rec", AQUA, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 1.75))
    x = np.arange(5); off = [-0.22, 0.0, 0.22]
    for ax, met, ylab in zip(axes, ["char_acc", "word_acc"], ["Character accuracy (1$-$CER)", "Word accuracy (1$-$WER)"]):
        ymax = 0
        for j, (m, lab, c, mk) in enumerate(meths):
            for i, s in enumerate(SCEN):
                v = [cidx[k][m][met] for k in keys if k[0] == s]
                a, b = ci(v); ymax = max(ymax, b)
                _point_interval(ax, x[i] + off[j], np.mean(v), a, b, c, mk, label=lab if i == 0 else None)
        if met == "char_acc":
            for i, s in enumerate(SCEN):
                ch = np.mean([ctrl[k]["Chance"]["char_acc"] for k in keys if k[0] == s])
                ax.hlines(ch, i - 0.38, i + 0.38, color=INK, lw=0.9, linestyles=(0, (3, 2)), zorder=2,
                          label="Chance (length-matched random text)" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(DEG_LONG)
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(0, ymax * 1.06)
        ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
        for i in range(1, 5):
            ax.axvline(i - 0.5, color=GRID, lw=0.5, zorder=1)
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, handlelength=1.6, columnspacing=1.6)
    fig.tight_layout(pad=0.2, w_pad=1.5)
    fig.savefig(out / "fig_ocr_scenario_v2.pdf"); plt.close(fig)


def fig_controls(ctrl, out):
    keys = sorted(ctrl)
    meths = [("DownUp-4x@384", "4$\\times$ down–up @384 px", INK2, "D"),
             ("DownUp-4x@full", "4$\\times$ down–up @full res.", INK2, "d"),
             ("Unsharp-0.5", "Unsharp mask, $\\alpha$=0.5", BLUE, "o"),
             ("Unsharp-1.2", "Unsharp mask, $\\alpha$=1.2", BLUE, "o"),
             ("Fusion", "Two-view fusion", ORANGE, "s")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.45), sharey=True)
    y = np.arange(len(meths))[::-1]
    for ax, met, lab in zip(axes, ["ssim", "char_acc", "word_acc"],
                            ["$\\Delta$SSIM vs. input", "$\\Delta$ character accuracy vs. input", "$\\Delta$ word accuracy vs. input"]):
        lo_all, hi_all = [0], [0]
        for yi, (m, ml, c, mk) in zip(y, meths):
            d = np.array([ctrl[k][m][met] - ctrl[k]["Input"][met] for k in keys])
            a, b = ci(d); lo_all.append(a); hi_all.append(b)
            ax.hlines(yi, a, b, color=c, lw=1.2, zorder=3)
            ax.plot(d.mean(), yi, marker=mk, ms=4, color=c, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.axvline(0, color=INK, lw=0.7, zorder=2)
        span = max(hi_all) - min(lo_all); ax.set_xlim(min(lo_all) - 0.06 * span, max(hi_all) + 0.06 * span)
        ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        ax.locator_params(axis="x", nbins=5)
    axes[0].set_yticks(y); axes[0].set_yticklabels([ml for _, ml, _, _ in meths])
    axes[0].set_ylim(-0.6, len(meths) - 0.4)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    fig.savefig(out / "fig_controls_v2.pdf"); plt.close(fig)


def fig_runtime(bench, out):
    st = bench["stages"]
    labels = ["S1 registration", "S2 dense flow", "S3–S4 weighting + fusion", "S5 post-processing$^{*}$"]
    keys = ["s2_registration_ms", "s3_flow_ms", "s4s5_fusion_ms", "s5_post_ms"]
    vals = [st[k]["mean"] for k in keys]; sds = [st[k]["std"] for k in keys]
    fig, ax = plt.subplots(figsize=(COLW, 1.25))
    y = np.arange(4)
    bars = ax.barh(y, vals, height=0.6, color=[BLUE, BLUE, BLUE, "white"], edgecolor=[BLUE, BLUE, BLUE, MUTED],
                   linewidth=[0, 0, 0, 0.8], zorder=3)
    bars[3].set_hatch("////")
    for i, (v, s) in enumerate(zip(vals, sds)):
        ax.text(v + 25, i, f"{v:.0f} $\\pm$ {s:.0f} ms", va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0, 1450); ax.set_xticks([0, 250, 500, 750, 1000, 1250])
    ax.set_xlabel("Latency (ms), mean $\\pm$ s.d. over 10 runs")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "fig_runtime_v2.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pics", default="../Pics")
    a = ap.parse_args(); out = Path(a.pics); out.mkdir(exist_ok=True)
    idx = load_idx(json.load(open("results.json"))["records"])
    cidx = load_idx(json.load(open("compare_proc512.json"))["records"])
    ctrl = load_idx(json.load(open("controls_512.json"))["records"] + json.load(open("fusion_512.json"))["records"])
    fig_alignment(idx, out); fig_tradeoff(idx, out); fig_ocr_scenario(cidx, ctrl, out)
    fig_controls(ctrl, out); fig_runtime(json.load(open("runtime_colab.json")), out)
    print("done")


if __name__ == "__main__":
    main()


def fig_qualitative(ds, out, pairs=(("S1", "pair_05"), ("S4", "pair_02"))):
    """Crops: input / unsharp / two-view fusion / ground truth (same crops as v1)."""
    import sys, cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import enhance_core as ec
    from controls_eval import unsharp
    names = {"S1": "D1 (Gaussian blur, $\\sigma$ = 2.5)", "S4": "D4 (motion blur + JPEG)"}
    rows = []
    for s, p in pairs:
        d = ds / s / p
        m1, m2, gt = (cv2.imread(str(d / f)) for f in ("m1.jpg", "m2.jpg", "gt.jpg"))
        r = ec.enhance(m1, m2, matcher="sift", multiscale=False, guided=False, use_post=False)
        lab = f"{names[s]}, page {int(p.split('_')[1])}; SIFT registration, {r['n_inliers']} RANSAC inliers"
        rows.append((lab, [m1, unsharp(m1, 1.2), r["result"], gt], r["roi"]))
    titles = ["Degraded input", "Single-image unsharp mask ($\\alpha$ = 1.2)", "Two-view fusion", "Ground truth"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(TEXTW, 1.62 * len(rows)))
    for i, (lab, imgs, roi) in enumerate(rows):
        x1, y1, x2, y2 = roi if roi else (0, 0, imgs[0].shape[1], imgs[0].shape[0])
        cw, chh = int(0.40 * (x2 - x1)), int(0.20 * (y2 - y1))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bx = (cx - cw // 2, cy - chh // 2, cx + cw // 2, cy + chh // 2)
        for j, im in enumerate(imgs):
            crop = im[bx[1]:bx[3], bx[0]:bx[2]]
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(MUTED); sp.set_linewidth(0.5)
            if i == 0:
                ax.set_title(titles[j], fontsize=7.5, pad=3)
        axes[i, 0].text(0.0, -0.06, lab, transform=axes[i, 0].transAxes, fontsize=7, va="top", ha="left", color=INK2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.07, wspace=0.03, hspace=0.22)
    fig.savefig(out / "fig_qualitative_v2.png", dpi=300); plt.close(fig)

"""fig_qual_v3.py -- qualitative comparison with OCR read-out (Fig. 6).
Columns: degraded input | single-image unsharp mask | TTSR (trained RefSR
network, saved outputs in refsr_sr_out/) | two-view fusion | ground truth.
Under every crop: Tesseract output on that crop (same preprocessing as the
paper, ocr_downstream_v2._ocr_text).
Usage: python fig_qual_v3.py <dataset> <refsr_sr_out> <out.png>"""
import sys, textwrap, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvpr_figures_v2 as F
import matplotlib.pyplot as plt
import enhance_core as ec, ocr_downstream_v2 as ocr
from controls_eval import unsharp

PAIRS = [("S1", "pair_05", (190, 118, 610, 262), "D1 (Gaussian blur, $\\sigma$ = 2.5), text page"),
         ("S4", "pair_02", (290, 672, 890, 812), "D4 (motion blur + JPEG), calendar page")]
TITLES = ["Degraded input", "Unsharp mask (single image)", "TTSR (trained RefSR)", "Two-view fusion (ours)", "Ground truth"]

def main(ds, sr, out):
    ds, sr = Path(ds), Path(sr)
    fig = plt.figure(figsize=(F.TEXTW, 2.85))
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 0.46, 0.68, 0.40], left=0.004, right=0.996, top=0.945, bottom=0.0, wspace=0.03, hspace=0.06)
    for i, (s, p, bx, lab) in enumerate(PAIRS):
        d = ds / s / p
        m1, m2, gt = (cv2.imread(str(d / f)) for f in ("m1.jpg", "m2.jpg", "gt.jpg"))
        r = ec.enhance(m1, m2, matcher="sift", multiscale=False, guided=False, use_post=False)
        tt = cv2.imread(str(sr / f"{s}_{p}.png"))
        if tt.shape != m1.shape: tt = cv2.resize(tt, (m1.shape[1], m1.shape[0]))
        for j, im in enumerate([m1, unsharp(m1, 1.2), tt, r["result"], gt]):
            c = im[bx[1]:bx[3], bx[0]:bx[2]]
            ax = fig.add_subplot(gs[2 * i, j]); ax.imshow(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(F.MUTED); sp.set_linewidth(0.5)
            if i == 0: ax.set_title(TITLES[j], fontsize=7.5, pad=3)
            txt = ocr._ocr_text(c.copy()); print(s, p, TITLES[j], "|", txt)
            pre = "OCR: " if j == 0 else ""
            lines = textwrap.wrap(pre + txt, 32) or [pre + "(no text detected)"]
            if len(lines) > 3: lines = lines[:3]; lines[-1] = lines[-1][:29] + "..."
            tx = fig.add_subplot(gs[2 * i + 1, j]); tx.axis("off")
            tx.text(0.0, 0.97, "\n".join(lines), family="DejaVu Sans Mono", fontsize=4.6, va="top", ha="left",
                    color=F.INK if j == 4 else F.INK2, linespacing=1.15, transform=tx.transAxes)
            if j == 0:
                tx.text(0.0, 0.02 if i == 0 else 0.0, f"{lab}; SIFT registration, {r['n_inliers']} inliers", fontsize=6.5, va="bottom", ha="left", color=F.INK2, transform=tx.transAxes)
    fig.savefig(out, dpi=300); plt.close(fig)
if __name__ == "__main__": main(*sys.argv[1:4])

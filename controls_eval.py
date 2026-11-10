"""
controls_eval.py -- reviewer-requested control experiments (CVPR revision)
==========================================================================
Single-image and protocol controls that isolate (i) what the TTSR input
protocol alone costs in fidelity and (ii) what a second view adds beyond
single-image sharpening.  Everything is scored on the FULL frame of the
degraded base M1 against the pixel-aligned ground truth, resized to a
common 512 px long side, with the SAME OCR preprocessing as
compare_headless.py.  No reference view is used by any control.

Methods
  Input             degraded M1
  DownUp-4x@384     the TTSR input round trip: resize to <=384 px long side,
                    bicubic /4, bicubic x4, bicubic back to full size
                    (exactly what refsr_ttsr_adapter.py does to M1)
  DownUp-4x@full    bicubic /4 and x4 at full resolution (no 384 cap)
  Unsharp-0.5       single-image unsharp mask, alpha=0.5, sigma=1.5
  Unsharp-1.2       single-image unsharp mask, alpha=1.2 (= S5 strength)
  Chance (OCR)      1-CER of a random string with the reference's length
                    and character distribution (20 draws per pair)

If pyiqa is importable (import it BEFORE this module) LPIPS/NIQE are
also computed; otherwise they are left out (SSIM/PSNR/OCR only).

Usage:  python controls_eval.py [dataset_dir] [out_json]
        python controls_eval.py dataset fusion_512.json --with-fusion
            (runs only the two-view fusion under the same protocol; the
             run is resumable, progress is cached in <out>.jsonl)
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_downstream_v2 as ocr          # same OCR code path as the paper
from jiwer import cer

try:
    import pyiqa, torch
    _LP = pyiqa.create_metric("lpips", device="cpu")
    _NQ = pyiqa.create_metric("niqe", device="cpu")
    HAVE_IQA = True
except Exception:
    HAVE_IQA = False

PROC = 512
MAX_DIM = 2000          # as in compare_headless.run(max_dim=2000)
TTSR_CAP = 384          # max_proc used for TTSR in the paper
SCALE = 4


def _resize_max(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    s = max_dim / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def _resize_long(img, proc):
    h, w = img.shape[:2]
    s = proc / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))))


def downup(m1, cap=None):
    H, W = m1.shape[:2]
    if cap:
        s = min(1.0, cap / max(H, W))
        Hp, Wp = int(H * s), int(W * s)
    else:
        Hp, Wp = H, W
    Hc, Wc = (Hp // SCALE) * SCALE, (Wp // SCALE) * SCALE
    x = cv2.resize(m1, (Wc, Hc))
    x = cv2.resize(x, (Wc // SCALE, Hc // SCALE), interpolation=cv2.INTER_CUBIC)
    x = cv2.resize(x, (Wc, Hc), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC)


def unsharp(img, alpha, sigma=1.5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + alpha, blur, -alpha, 0)


def score(res, gt):
    g1 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    out = dict(ssim=float(structural_similarity(g1, g2, data_range=255)),
               psnr=float(peak_signal_noise_ratio(g2, g1, data_range=255)),
               lapvar=float(cv2.Laplacian(g1, cv2.CV_64F).var()))
    if HAVE_IQA:
        def t(im):
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1)[None]
        with torch.no_grad():
            out["lpips"] = float(_LP(t(res), t(gt)))
            out["niqe"] = float(_NQ(t(res)))
    return out


def chance_char_acc(ref, rng, draws=20):
    chars = list(ref)
    vals = []
    for _ in range(draws):
        hyp = "".join(rng.choice(chars, size=len(chars)))
        vals.append(max(0.0, 1.0 - float(cer(ref, hyp))))
    return float(np.mean(vals))


def main(dataset="dataset", out="controls_512.json"):
    rng = np.random.default_rng(0)
    root = Path(dataset)
    methods = {
        "Input": lambda m, r: m,
        "DownUp-4x@384": lambda m, r: downup(m, TTSR_CAP),
        "DownUp-4x@full": lambda m, r: downup(m, None),
        "Unsharp-0.5": lambda m, r: unsharp(m, 0.5),
        "Unsharp-1.2": lambda m, r: unsharp(m, 1.2),
    }
    # optional: the two-view fusion itself (single-scale, no post) under the
    # SAME full-frame protocol, so the second view can be compared against
    # single-image sharpening on identical pairs.  matcher='auto' uses
    # LightGlue if installed, else SIFT (the matcher used is recorded).
    if "--with-fusion" in sys.argv:
        import enhance_core as ec
        def _fusion(m, r):
            res = ec.enhance(m, r, matcher="auto", multiscale=False, guided=False, use_post=False)
            _fusion.last = (res["aligned"], res["matcher_used"], res["n_inliers"])
            return res["result"]
        methods = {"Fusion": _fusion}
        sys.argv.remove("--with-fusion")
    # resumable: per-pair records are appended to <out>.jsonl
    cache = Path(out).with_suffix(".jsonl")
    records, done = [], set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line); records.append(r); done.add((r["scenario"], r["pair"]))
    for pdir in sorted(p for p in root.glob("S*/pair_*") if p.is_dir()):
        if (pdir.parent.name, pdir.name) in done:
            continue
        new = []
        m1 = _resize_max(cv2.imread(str(pdir / "m1.jpg")), MAX_DIM)
        gt = cv2.imread(str(pdir / "gt.jpg"))
        if gt.shape != m1.shape:
            gt = cv2.resize(gt, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_AREA)
        gt_p = _resize_long(gt, PROC)
        ref_text = " ".join((pdir / "gt.txt").read_text(encoding="utf-8").split()).strip().lower()
        if len(ref_text) < ocr.MIN_REF_CHARS:
            ref_text = None
        m2 = cv2.imread(str(pdir / "m2.jpg"))
        for name, fn in methods.items():
            img = _resize_long(fn(m1, m2), PROC)
            rec = score(img, gt_p)
            rec.update(scenario=pdir.parent.name, pair=pdir.name, method=name)
            if name == "Fusion":
                rec["aligned"], rec["matcher"], rec["n_inliers"] = fn.last
            if ref_text:
                ca, wa = ocr._accuracy(ref_text, ocr._ocr_text(img.copy()))
                rec["char_acc"], rec["word_acc"] = ca, wa
            new.append(rec)
        if ref_text and "Input" in methods:
            new.append(dict(scenario=pdir.parent.name, pair=pdir.name, method="Chance",
                            char_acc=chance_char_acc(ref_text, rng), word_acc=0.0,
                            ref_len=len(ref_text)))
        with open(cache, "a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
        records.extend(new)
        print(pdir, "done", flush=True)

    # summary
    names = sorted({r["method"] for r in records})
    summary = {}
    for m in names:
        rows = [r for r in records if r["method"] == m]
        summary[m] = {}
        for k in ["ssim", "psnr", "lapvar", "lpips", "niqe", "char_acc", "word_acc"]:
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                summary[m][k] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), n=len(v))
    json.dump(dict(records=records, summary=summary, proc=PROC, have_iqa=HAVE_IQA),
              open(out, "w"), indent=1)
    print(f"\n{'Method':16s}{'SSIM':>8s}{'PSNR':>8s}{'LapVar':>9s}{'OCRc':>8s}{'OCRw':>8s}")
    for m in names:
        s = summary[m]
        f = lambda k, p=3: f"{s[k]['mean']:.{p}f}" if k in s else "--"
        print(f"{m:16s}{f('ssim'):>8s}{f('psnr',2):>8s}{f('lapvar',1):>9s}{f('char_acc'):>8s}{f('word_acc'):>8s}")
    print("saved", out)


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("--")])

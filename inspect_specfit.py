#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_specfit.py
==================
查看 specfit.py 输出的参数 FITS

功能
----
- 列出所有 HDU + 每个参数的统计量（有效像素数 / min / median / max）
- --plot：一次画出所有参数的网格面板大图
- --compare truth.fits：按 HDU 名对比真值，输出残差统计 + 残差图

用法
----
    python inspect_specfit.py --input specfit_params.fits
    python inspect_specfit.py --input specfit_params.fits --plot params_panel.png
    python inspect_specfit.py --input specfit_params.fits \\
        --compare mock_cube_truth.fits --plot params_panel.png
"""

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[错误] 需要 matplotlib：pip install matplotlib", file=sys.stderr)
    sys.exit(2)

try:
    from astropy.io import fits
except ImportError:
    print("[错误] 需要 astropy：pip install astropy", file=sys.stderr)
    sys.exit(2)


# ==================== 工具 ====================
def friendly_name(name):
    n = name.strip().lower()
    if n == "chi2":
        return "Reduced χ²"
    if n == "ncomp":
        return "N_components"
    if n.startswith("amp") and n[3:].isdigit():
        return f"Amplitude {n[3:]}"
    if n.startswith("sigma") and n[5:].isdigit():
        return f"FWHM {n[5:]}"
    if n.startswith("v") and n[1:].isdigit():
        return f"Velocity {n[1:]}"
    if n == "c0":
        return "Baseline c0"
    if n == "c1":
        return "Baseline c1"
    if n.endswith("_err"):
        return friendly_name(n[:-4]) + " err"
    return name


def load_hdus(path):
    """返回 ([(name, data, header), ...], primary_header)。"""
    if not os.path.exists(path):
        raise ValueError(f"文件不存在：{path}")
    items = []
    with fits.open(path) as hdul:
        primary = hdul[0].header.copy()
        for h in hdul:
            if h.data is None:
                continue
            d = np.asarray(h.data)
            if d.ndim != 2:
                continue
            items.append((h.name, d, h.header.copy()))
    if not items:
        raise ValueError(f"{path}: 无 2D 图像 HDU")
    return items, primary


def stat_summary(name, d):
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return f"  {name:<12s}  全部 NaN"
    return (f"  {name:<12s}  valid={finite.size:>8d}/{d.size:<8d}  "
            f"min={finite.min():>11.4g}  med={np.median(finite):>11.4g}  "
            f"max={finite.max():>11.4g}")


# ==================== 列表 + 统计 ====================
def print_listing(items, primary):
    print(f"\n== 文件信息 ==")
    for key in ("NSCOMP", "BASELINE", "NOISE", "AUTO_N", "MAX_NC",
                "OBJECT", "BUNIT"):
        if key in primary:
            print(f"  {key:<10s} = {primary[key]}")
    print(f"\n== HDU 统计 ==")
    for name, data, _ in items:
        print(stat_summary(name, data))


# ==================== 网格面板 ====================
def plot_panel(items, out_png, cmap, dpi, show_err=False):
    # 过滤掉 _err（除非指定）和纯 NaN
    show = []
    for name, data, _ in items:
        if not show_err and name.endswith("_err"):
            continue
        if not np.any(np.isfinite(data)):
            continue
        show.append((name, data))
    if not show:
        print("[警告] 无可画的 HDU")
        return

    n = len(show)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.8 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).flatten()

    for i, (name, data) in enumerate(show):
        ax = axes[i]
        d = data.astype(float)

        # sigma -> FWHM
        if name.lower().startswith("sigma"):
            d = d * 2.3548

        masked = np.ma.masked_invalid(d)
        is_int = np.issubdtype(data.dtype, np.integer)

        if is_int:
            im = ax.imshow(masked, origin="lower", cmap=cmap,
                           interpolation="nearest")
        else:
            vmin = np.nanpercentile(masked.compressed(), 1) if masked.count() else None
            vmax = np.nanpercentile(masked.compressed(), 99) if masked.count() else None
            im = ax.imshow(masked, origin="lower", cmap=cmap,
                           interpolation="nearest", vmin=vmin, vmax=vmax)

        ax.set_title(f"{friendly_name(name)}  [{name}]", fontsize=9)
        ax.set_xlabel("x [pix]", fontsize=8)
        ax.set_ylabel("y [pix]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     format="%.3g" if not is_int else "%d")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("specfit parameters", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_png}")


# ==================== 对比 truth ====================
# truth 里 HDU 名 vs fit 里 HDU 名的映射
TRUTH_MAP = {
    "AMP": "amp",
    "V":   "v",
    "SIGMA": "sigma",
    "NCOMP": "ncomp",
}


def _parse_truth_name(name):
    """'AMP1' -> ('amp', 1); 'NCOMP' -> ('ncomp', None); 'V2' -> ('v', 2)。"""
    n = name.strip().upper()
    if n == "NCOMP":
        return "ncomp", None
    for k, v in TRUTH_MAP.items():
        if n.startswith(k):
            rest = n[len(k):]
            if rest.isdigit():
                return v, int(rest)
            return v, None
    return name.lower(), None


def compare_with_truth(fit_items, truth_path, plot_path, cmap, dpi):
    if not os.path.exists(truth_path):
        print(f"[错误] truth 文件不存在：{truth_path}", file=sys.stderr)
        return

    truth_map = {}
    with fits.open(truth_path) as hdul:
        for h in hdul:
            if h.data is None or h.data.ndim != 2:
                continue
            kind, k = _parse_truth_name(h.name)
            truth_map[(kind, k)] = np.asarray(h.data, dtype=float)

    if not truth_map:
        print(f"[错误] {truth_path} 没有 2D 图像 HDU", file=sys.stderr)
        return

    print(f"\n== 与真值对比 ({os.path.basename(truth_path)}) ==")
    diffs = []
    for name, data, _ in fit_items:
        if name.endswith("_err"):
            continue
        if name == "chi2":
            continue
        # fit HDU 名 → kind + k
        fit_name = name.lower()
        kind, k = None, None
        for base in ("amp", "sigma", "v", "c"):
            if fit_name.startswith(base):
                rest = fit_name[len(base):]
                if rest.isdigit():
                    kind, k = base, int(rest)
                elif rest == "":
                    kind, k = base, None
                break
        if kind is None:
            continue
        if kind == "sigma":
            key = ("sigma", k)
        elif kind == "amp":
            key = ("amp", k)
        elif kind == "v":
            key = ("v", k)
        else:
            key = ("c", k)
        if key not in truth_map:
            continue

        fit_arr = np.asarray(data, dtype=float)
        truth_arr = truth_map[key]
        if fit_arr.shape != truth_arr.shape:
            print(f"  [跳过] {name}: shape 不一致 "
                  f"{fit_arr.shape} vs {truth_arr.shape}")
            continue

        valid = np.isfinite(fit_arr) & np.isfinite(truth_arr)
        # 忽略 truth 里 amp=0（该成分不存在）的像素
        if kind == "amp":
            valid &= (truth_arr != 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        diff = fit_arr[valid] - truth_arr[valid]
        rel = diff / np.where(np.abs(truth_arr[valid]) > 1e-12,
                              truth_arr[valid], np.nan)
        print(f"  {name:<10s}  n={n_valid:>6d}  "
              f"bias={np.nanmean(diff):>10.4g}  "
              f"rms={np.sqrt(np.nanmean(diff**2)):>10.4g}  "
              f"med_rel_err={np.nanmedian(np.abs(rel)):>8.2%}")

        diffs.append((name, fit_arr, truth_arr, valid))

    if not diffs or not plot_path:
        return

    # 画 残差 = fit - truth
    n = len(diffs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.8 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).flatten()

    for i, (name, fit_arr, truth_arr, valid) in enumerate(diffs):
        ax = axes[i]
        resid = np.full_like(fit_arr, np.nan, dtype=float)
        resid[valid] = fit_arr[valid] - truth_arr[valid]
        masked = np.ma.masked_invalid(resid)
        vm = np.nanpercentile(np.abs(masked.compressed()), 98) if masked.count() else 1
        im = ax.imshow(masked, origin="lower", cmap="RdBu_r",
                       vmin=-vm, vmax=vm, interpolation="nearest")
        ax.set_title(f"residual {name}", fontsize=9)
        ax.set_xlabel("x [pix]", fontsize=8)
        ax.set_ylabel("y [pix]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2g")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("fit − truth  residual", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {plot_path}")


# ==================== CLI ====================
def build_parser():
    p = argparse.ArgumentParser(
        description="查看 specfit.py 输出的参数 FITS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python inspect_specfit.py --input specfit_params.fits
  python inspect_specfit.py --input specfit_params.fits --plot panel.png
  python inspect_specfit.py --input specfit_params.fits \\
      --compare mock_cube_truth.fits --plot panel.png
""")
    p.add_argument("--input", required=True, help="specfit 参数 FITS")
    p.add_argument("--plot", default=None,
                   help="输出参数网格 PNG（留空=只打印统计）")
    p.add_argument("--compare", default=None,
                   help="与 truth FITS 对比，输出残差统计 + 残差图")
    p.add_argument("--show-err", action="store_true",
                   help="同时画 *_err 误差 HDU")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=150)
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    try:
        items, primary = load_hdus(args.input)
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    print_listing(items, primary)

    if args.plot:
        plot_panel(items, args.plot, args.cmap, args.dpi,
                   show_err=args.show_err)

    if args.compare:
        resid_png = None
        if args.plot:
            stem, _ = os.path.splitext(args.plot)
            resid_png = f"{stem}_residual.png"
        compare_with_truth(items, args.compare, resid_png,
                           args.cmap, args.dpi)

    return 0


if __name__ == "__main__":
    sys.exit(main())

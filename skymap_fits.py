#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skymap_fits.py
==============
FITS 天图绘制工具（多文件 / 多 HDU / 自动 WCS / 可选球面投影）

依赖：numpy, matplotlib, astropy

投影
----
native    使用 FITS 自带的 WCS 投影（默认，直角坐标）
rect      矩形经纬坐标（可重新居中）
aitoff    Aitoff / Hammer-Aitoff 全天投影
hammer    Hammer 等面积投影
mollweide Mollweide 等面积投影
lambert   Lambert 方位等面积投影

主要用法
--------
FITS 原始投影（旧行为）：
    python skymap_fits.py --input HI4PI.fits --cmap viridis --log

全天 Aitoff 银道图（推荐用于柱密度等全天数据）：
    python skymap_fits.py --input HI4PI.fits --projection aitoff \\
        --coordsys galactic --log --cmap viridis

Mollweide 赤道图（以 RA=180 为中心）：
    python skymap_fits.py --input HI4PI.fits --projection mollweide \\
        --coordsys equatorial --center-lon 180 --log

多个 FITS：
    python skymap_fits.py --inputs U.fits,B.fits,V.fits,R.fits \\
        --labels U,B,V,R --projection aitoff --ncols 2

其他：
    python skymap_fits.py --input x.fits --list-hdu
    python skymap_fits.py --input cube.fits --slice 5 --projection aitoff
"""

import argparse
import os
import sys
import warnings

try:
    import numpy as np
except ImportError:
    print("[错误] 需要 numpy：pip install numpy", file=sys.stderr)
    sys.exit(2)

try:
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
except ImportError:
    print("[错误] 需要 matplotlib：pip install matplotlib", file=sys.stderr)
    sys.exit(2)

try:
    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    warnings.simplefilter("ignore", FITSFixedWarning)
except ImportError:
    print("[错误] 需要 astropy：pip install astropy", file=sys.stderr)
    sys.exit(2)


# ==================== 投影常量 ====================
MPL_PROJECTIONS = ("aitoff", "hammer", "mollweide", "lambert")
PROJECTION_CHOICES = ("native", "rect") + MPL_PROJECTIONS


# ==================== 小工具 ====================
def _split_csv(s):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _coords_label_from_wcs(wcs):
    if wcs is None:
        return "", ""
    try:
        ctype0 = (wcs.wcs.ctype[0] or "").upper()
        ctype1 = (wcs.wcs.ctype[1] or "").upper()
    except Exception:
        return "", ""
    if "GLON" in ctype0 or "GLAT" in ctype1:
        return "Galactic l [deg]", "Galactic b [deg]"
    if "RA" in ctype0 or "DEC" in ctype1:
        return "RA [deg]", "Dec [deg]"
    return wcs.wcs.ctype[0] or "x", wcs.wcs.ctype[1] or "y"


def _infer_coordsys(wcs):
    """从 WCS CTYPE 推断坐标系：'galactic' / 'equatorial'。"""
    if wcs is None:
        return "equatorial"
    try:
        ctype0 = (wcs.wcs.ctype[0] or "").upper()
    except Exception:
        return "equatorial"
    if "GLON" in ctype0 or "GLAT" in ctype0:
        return "galactic"
    return "equatorial"


# ==================== HDU 解析 ====================
def _resolve_hdu_indices(hdul, spec):
    s = (spec or "").strip().lower()
    if not s or s == "auto":
        for i, h in enumerate(hdul):
            if h.data is not None and h.data.ndim in (2, 3):
                return [i]
        raise ValueError("该文件中找不到 2D/3D 图像 HDU")
    if s == "all":
        idxs = [i for i, h in enumerate(hdul)
                if h.data is not None and h.data.ndim in (2, 3)]
        if not idxs:
            raise ValueError("该文件中找不到任何 2D/3D 图像 HDU")
        return idxs
    idxs = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idxs.append(int(tok))
        except ValueError:
            raise ValueError(f"无效 HDU 索引：{tok!r}")
    if not idxs:
        raise ValueError(f"无效 --hdu 参数：{spec!r}")
    return idxs


def _pick_wcs(header):
    try:
        w = WCS(header)
    except Exception:
        return None
    try:
        if w.has_celestial:
            return w.celestial
    except Exception:
        pass
    return None


# ==================== 加载 ====================
def load_one_hdu(hdul, idx, slice_index, max_pixels):
    hdu = hdul[idx]
    if hdu.data is None:
        raise ValueError(f"HDU[{idx}] 无数据")
    data = hdu.data
    if data.ndim < 2:
        raise ValueError(f"HDU[{idx}] 是 {data.ndim}D，需要至少 2D")

    header = hdu.header.copy()
    if header.get("PIXTYPE", "").upper() == "HEALPIX":
        raise ValueError(
            f"HDU[{idx}]: HEALPix 格式不支持直接绘制，"
            f"请先用 healpy/reproject 转成 WCS 图像")

    if data.ndim == 3:
        if slice_index is None:
            raise ValueError(
                f"HDU[{idx}] 是 3D cube（shape={data.shape}），"
                f"请用 --slice 选择切片")
        if not (0 <= slice_index < data.shape[0]):
            raise ValueError(f"--slice {slice_index} 超出 [0, {data.shape[0]})")
        data = data[slice_index]
    elif data.ndim > 3:
        raise ValueError(f"HDU[{idx}] 是 {data.ndim}D，暂不支持")

    data = np.asarray(data, dtype=float)
    wcs = _pick_wcs(header)

    if max_pixels and max_pixels > 0:
        ny, nx = data.shape
        stride = max(1, max(ny, nx) // max_pixels)
        if stride > 1:
            print(f"[提示] HDU[{idx}] 降采样 stride={stride}", file=sys.stderr)
            data = data[::stride, ::stride]
            if wcs is not None:
                try:
                    wcs = wcs.deepcopy()
                    wcs.wcs.crpix[0] = (wcs.wcs.crpix[0] - 0.5) / stride + 0.5
                    wcs.wcs.crpix[1] = (wcs.wcs.crpix[1] - 0.5) / stride + 0.5
                    wcs.wcs.cdelt[0] *= stride
                    wcs.wcs.cdelt[1] *= stride
                except Exception:
                    pass

    unit = header.get("BUNIT", "").strip()
    hdu_name = (hdu.name or f"HDU{idx}").strip()

    return dict(data=data, header=header, wcs=wcs, unit=unit,
                hdu_index=idx, hdu_name=hdu_name)


def load_inputs(args):
    files = _split_csv(args.input) + _split_csv(args.inputs)
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]
    if not files:
        raise ValueError("请用 --input 或 --inputs 指定 FITS 文件")

    items = []
    for path in files:
        if not os.path.exists(path):
            raise ValueError(f"文件不存在：{path}")
        with fits.open(path) as hdul:
            idxs = _resolve_hdu_indices(hdul, args.hdu)
            base = os.path.splitext(os.path.basename(path))[0]
            for idx in idxs:
                item = load_one_hdu(hdul, idx, args.slice, args.max_pixels)
                if len(idxs) == 1:
                    lbl = base
                else:
                    lbl = f"{base} [{item['hdu_name']}]"
                item["label"] = lbl
                item["path"] = path
                items.append(item)

    if args.labels:
        lbls = _split_csv(args.labels)
        if len(lbls) == len(items):
            for it, l in zip(items, lbls):
                it["label"] = l
        else:
            print(f"[警告] --labels 给 {len(lbls)} 个，但只 {len(items)} 个面板，忽略",
                  file=sys.stderr)
    return items


# ==================== 归一化 ====================
def _parse_percentile(s):
    if not s:
        return None
    parts = _split_csv(s)
    if len(parts) != 2:
        raise ValueError(f"--percentile 需要 'p_lo,p_hi'，收到 {s!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0 <= lo < hi <= 100):
        raise ValueError("--percentile 范围应在 [0,100] 且 p_lo<p_hi")
    return lo, hi


def _compute_norm(data, args):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("数据全是 NaN/Inf，无法归一化")

    percentile = _parse_percentile(args.percentile)
    if percentile:
        vmin = float(np.percentile(finite, percentile[0]))
        vmax = float(np.percentile(finite, percentile[1]))
    else:
        vmin, vmax = float(finite.min()), float(finite.max())

    if args.vmin is not None:
        vmin = args.vmin
    if args.vmax is not None:
        vmax = args.vmax

    if args.log:
        pos = finite[finite > 0]
        if pos.size == 0:
            raise ValueError("对数拉伸需要正数据")
        if vmin <= 0:
            vmin = float(np.percentile(pos, 0.5))
        if vmax <= vmin:
            vmax = float(pos.max())
        if vmax <= vmin:
            raise ValueError("无法确定对数拉伸范围")
        return colors.LogNorm(vmin=vmin, vmax=vmax)

    if vmax <= vmin:
        raise ValueError(f"非法范围 vmin={vmin} vmax={vmax}")
    return colors.Normalize(vmin=vmin, vmax=vmax)


# ==================== 球面投影辅助 ====================
def _pixel_lonlat(data_shape, wcs, target_cs):
    """返回每个像素中心的 lon/lat（度），坐标系 = target_cs。"""
    ny, nx = data_shape
    xx = np.arange(nx, dtype=float) + 0.5
    yy = np.arange(ny, dtype=float) + 0.5
    XX, YY = np.meshgrid(xx, yy)

    if wcs is None:
        raise ValueError("无 WCS，无法计算球面坐标")

    try:
        world = wcs.all_pix2world(XX, YY, 0)
    except Exception as e:
        raise ValueError(f"WCS 转换失败：{e}")
    lon_in, lat_in = world[0], world[1]

    input_cs = _infer_coordsys(wcs)
    if input_cs == target_cs:
        return lon_in, lat_in

    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        raise ValueError("坐标系转换需要 astropy.coordinates")

    if input_cs == "galactic":
        sc = SkyCoord(l=lon_in * u.deg, b=lat_in * u.deg, frame="galactic")
    else:
        sc = SkyCoord(ra=lon_in * u.deg, dec=lat_in * u.deg, frame="icrs")

    if target_cs == "galactic":
        sc = sc.galactic
        return sc.l.deg, sc.b.deg
    else:
        sc = sc.icrs
        return sc.ra.deg, sc.dec.deg


def _center_and_wrap(lon, center):
    """把 lon 居中到 [-180, 180)。"""
    return (lon - center + 180.0) % 360.0 - 180.0


def _check_skyspan(lon, lat):
    """返回 (lon_span_deg, lat_span_deg)，用于检测是否是全天图。"""
    finite_lon = lon[np.isfinite(lon)]
    finite_lat = lat[np.isfinite(lat)]
    if finite_lon.size == 0 or finite_lat.size == 0:
        return 0.0, 0.0
    lon_wrapped = np.sort(finite_lon % 360.0)
    gaps = np.diff(np.concatenate([lon_wrapped, [lon_wrapped[0] + 360.0]]))
    max_gap = gaps.max()
    lon_span = 360.0 - max_gap
    lat_span = finite_lat.max() - finite_lat.min()
    return float(lon_span), float(lat_span)


def _downsample_grid(lon, lat, data, max_side):
    ny, nx = data.shape
    if max_side <= 0 or (nx <= max_side and ny <= max_side):
        return lon, lat, data
    stride = max(1, max(nx // max_side, ny // max_side))
    return lon[::stride, ::stride], lat[::stride, ::stride], data[::stride, ::stride]


def _mask_wrap_jumps(lon_deg, data, threshold_deg=170.0):
    """在经度 wrap 处把数据设为 NaN，避免 pcolormesh 画出跨越整图的伪线。

    lon_deg 是已经 wrap 到 [-180, 180) 的经度（度）。
    """
    data = data.copy()
    if lon_deg.ndim != 2 or data.ndim != 2:
        return data
    dlon = np.diff(lon_deg, axis=1)                       # (ny, nx-1)
    bad = np.abs(dlon) > threshold_deg                    # wrap 处 |Δlon| ≈ 360°
    bad_full = np.zeros_like(lon_deg, dtype=bool)
    bad_full[:, :-1] |= bad
    bad_full[:, 1:] |= bad
    data[bad_full] = np.nan
    return data


# ==================== native 面板 ====================
def _draw_native_panel(fig, ax, item, args):
    data = item["data"]
    wcs = item["wcs"]
    norm = _compute_norm(data, args)

    im = ax.imshow(
        data, origin=args.origin, cmap=args.cmap, norm=norm,
        aspect="auto", interpolation="nearest", rasterized=True,
    )

    if wcs is not None:
        try:
            lon, lat = ax.coords
            ll, bb = _coords_label_from_wcs(wcs)
            lon.set_axislabel(ll, fontsize=9)
            lat.set_axislabel(bb, fontsize=9)
            lon.set_major_formatter("d.dd")
            lat.set_major_formatter("d.dd")
            ax.grid(color="white", alpha=0.3, linestyle="--")
        except Exception:
            ax.set_xlabel("x [pix]", fontsize=9)
            ax.set_ylabel("y [pix]", fontsize=9)
    else:
        ax.set_xlabel("x [pix]", fontsize=9)
        ax.set_ylabel("y [pix]", fontsize=9)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(item.get("unit") or "value", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    title = item["label"]
    if item.get("hdu_name") and args.show_hdu:
        title = f"{title}  [{item['hdu_name']}]"
    ax.set_title(title, fontsize=10, pad=8)


# ==================== 球面 / 矩形投影面板 ====================
def _draw_projected_panel(fig, position, item, args, projection):
    """用 matplotlib 球面投影（或矩形经纬轴）绘制单个面板。"""
    data = item["data"]
    wcs = item["wcs"]

    # 目标坐标系
    if args.coordsys == "auto":
        target_cs = _infer_coordsys(wcs)
    else:
        target_cs = args.coordsys

    # 每像素 lon/lat（度）
    lon, lat = _pixel_lonlat(data.shape, wcs, target_cs)

    # 中心经度
    center = args.center_lon
    if center is None:
        center = 0.0 if target_cs == "galactic" else 180.0
    lon_rel = _center_and_wrap(lon, center)

    # 非全天图提醒
    span_lon, span_lat = _check_skyspan(lon, lat)
    if projection != "rect" and (span_lon < 350.0 or span_lat < 170.0):
        print(f"[警告] {item['label']}: 天区跨度 lon={span_lon:.0f}° "
              f"lat={span_lat:.0f}°，不是全天图；{projection} 投影下会很小。"
              f"建议 --projection rect 或 native", file=sys.stderr)

    # 降采样
    if projection != "rect":
        lon_rel, lat, data = _downsample_grid(
            lon_rel, lat, data, args.max_spherical_pixels)

    # 断开 wrap 处的伪线
    data = _mask_wrap_jumps(lon_rel, data)

    norm = _compute_norm(data, args)

    # 建轴：*position 解包成 (nrows, ncols, index)
    if projection == "rect":
        ax = fig.add_subplot(*position)
        x_plot = lon_rel
        y_plot = lat
    else:
        ax = fig.add_subplot(*position, projection=projection)
        x_plot = np.radians(lon_rel)
        y_plot = np.radians(lat)

    # 抑制 pcolormesh 对 wrap 处非单调坐标的警告（数据已 mask，结果正确）
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*cell centers.*",
            category=UserWarning,
        )
        im = ax.pcolormesh(x_plot, y_plot, data, shading="auto",
                           cmap=args.cmap, norm=norm, rasterized=True)

    # 网格（getattr 兜底，兼容老 parser）
    if not getattr(args, "no_grid", False):
        try:
            ax.grid(True, alpha=0.3, linestyle="--")
        except Exception:
            pass

    if projection == "rect":
        ll = "l [deg]" if target_cs == "galactic" else "RA [deg]"
        bb = "b [deg]" if target_cs == "galactic" else "Dec [deg]"
        ax.set_xlabel(ll, fontsize=9)
        ax.set_ylabel(bb, fontsize=9)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal", adjustable="box")

    # colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(item.get("unit") or "value", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # 标题
    title = item["label"]
    if item.get("hdu_name") and args.show_hdu:
        title = f"{title}  [{item['hdu_name']}]"
    if target_cs == "galactic":
        suffix = f" (l/b, center={center:.0f}°)"
    else:
        suffix = f" (RA/Dec, center={center:.0f}°)"
    ax.set_title(f"{title}{suffix}", fontsize=10, pad=8)


# ==================== 主绘图入口 ====================
def plot_items(items, args):
    n = len(items)
    ncols = args.ncols if args.ncols > 0 else min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(
        figsize=(args.figsize[0] * ncols, args.figsize[1] * nrows),
        facecolor="white",
    )

    projection = args.projection
    for i, item in enumerate(items):
        position = (nrows, ncols, i + 1)
        if projection == "native":
            wcs = item["wcs"]
            if wcs is not None:
                try:
                    ax = fig.add_subplot(*position, projection=wcs)
                except Exception:
                    ax = fig.add_subplot(*position)
            else:
                ax = fig.add_subplot(*position)
            _draw_native_panel(fig, ax, item, args)
        else:
            _draw_projected_panel(fig, position, item, args, projection)

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ==================== --list-hdu ====================
def print_hdu_info(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"[跳过] {path} 不存在", file=sys.stderr)
            continue
        print(f"\n=== {path} ===")
        try:
            with fits.open(path) as hdul:
                for i, h in enumerate(hdul):
                    shape = getattr(h.data, "shape", None) if h.data is not None else None
                    dtype = getattr(h.data, "dtype", "") if h.data is not None else ""
                    bunit = h.header.get("BUNIT", "")
                    ctype = h.header.get("CTYPE1", "")
                    print(f"  HDU[{i}] name={h.name!r:12s} "
                          f"shape={str(shape):<20s} dtype={str(dtype):<10s} "
                          f"BUNIT={bunit!r}  CTYPE1={ctype!r}")
        except Exception as e:
            print(f"  [错误] {e}", file=sys.stderr)


# ==================== CLI ====================
_NEG_VALUE_OPTS = {"--vmin", "--vmax", "--slice", "--center-lon"}


def _normalize_argv(argv):
    out, i, n = [], 0, len(argv)
    while i < n:
        tok = argv[i]
        if tok in _NEG_VALUE_OPTS and i + 1 < n:
            nxt = argv[i + 1]
            if nxt.startswith("-") and not nxt.startswith("--"):
                out.append(f"{tok}={nxt}")
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="FITS 天图绘制（多文件 / 多 HDU / WCS / 可选球面投影）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python skymap_fits.py --input HI4PI.fits --cmap viridis --log
  python skymap_fits.py --input HI4PI.fits --projection aitoff --coordsys galactic --log
  python skymap_fits.py --input HI4PI.fits --projection mollweide \\
      --coordsys equatorial --center-lon 180 --log
  python skymap_fits.py --input x.fits --projection rect --center-lon 0
  python skymap_fits.py --inputs a.fits,b.fits --labels A,B --projection aitoff --ncols 2
  python skymap_fits.py --input x.fits --list-hdu
""")

    src = p.add_argument_group("数据源")
    src.add_argument("--input", default=None,
                     help="FITS 路径（支持逗号分隔多文件）")
    src.add_argument("--inputs", default=None,
                     help="--input 的别名（可与 --input 组合）")
    src.add_argument("--hdu", default=None,
                     help="HDU 索引：数字 / 逗号分隔 / all（默认 auto）")
    src.add_argument("--slice", type=int, default=None,
                     help="3D cube 切片索引（沿第一轴）")
    src.add_argument("--max-pixels", type=int, default=0,
                     help="超过该尺寸时降采样（0=不降采样）")
    src.add_argument("--list-hdu", action="store_true",
                     help="打印所有输入的 HDU 结构后退出")

    plot = p.add_argument_group("绘图")
    plot.add_argument("--projection", default="native",
                      choices=list(PROJECTION_CHOICES),
                      help="投影方式：native(默认)=用FITS自带WCS；"
                           "rect=矩形经纬；aitoff/hammer/mollweide/lambert=球面投影")
    plot.add_argument("--coordsys", default="auto",
                      choices=["auto", "galactic", "equatorial"],
                      help="绘制坐标系（默认 auto=沿用输入 WCS）")
    plot.add_argument("--center-lon", type=float, default=None,
                      help="投影中心经度（度）。galactic 默认 0；equatorial 默认 180")
    plot.add_argument("--max-spherical-pixels", type=int, default=720,
                      help="球面投影时最大边长像素（超出降采样，默认 720）")
    plot.add_argument("--labels", default=None,
                      help="逗号分隔的面板标题（数量应与面板数一致）")
    plot.add_argument("--title", default="FITS Sky Map")
    plot.add_argument("--cmap", default="viridis", help="matplotlib colormap")
    plot.add_argument("--log", action="store_true",
                      help="对数归一化（颜色）")
    plot.add_argument("--percentile", default=None,
                      help="颜色范围取分位数，如 '1,99'")
    plot.add_argument("--vmin", type=float, default=None)
    plot.add_argument("--vmax", type=float, default=None)
    plot.add_argument("--origin", choices=["lower", "upper"], default="lower",
                      help="像素原点（FITS 规范为 lower）")
    plot.add_argument("--ncols", type=int, default=0,
                      help="子图列数（0=min(3, N)）")
    plot.add_argument("--figsize", nargs=2, type=float, default=(5.0, 4.2),
                      metavar=("W", "H"), help="单个子图尺寸（英寸）")
    plot.add_argument("--show-hdu", action="store_true",
                      help="标题中附加 HDU 名")
    plot.add_argument("--no-grid", action="store_true",
                      help="不显示经纬网格")

    out = p.add_argument_group("输出")
    out.add_argument("--output", default="skymap_fits.png")
    out.add_argument("--dpi", type=int, default=150)
    out.add_argument("--no-show", action="store_true",
                     help="不弹窗（GUI 调用时加此项）")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(argv)
    args = build_parser().parse_args(argv)

    files = _split_csv(args.input) + _split_csv(args.inputs)

    if args.list_hdu:
        if not files:
            print("[错误] --list-hdu 需要 --input/--inputs", file=sys.stderr)
            return 1
        print_hdu_info(files)
        return 0

    try:
        items = load_inputs(args)
        if not items:
            raise ValueError("没有可用图像 HDU")
        fig = plot_items(items, args)

        if args.output:
            fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                        pad_inches=0.2, facecolor=fig.get_facecolor())
            print(f"天图已保存到 {args.output}")

        if not args.no_show:
            try:
                plt.show()
            except Exception as e:
                print(f"[警告] 无法显示：{e}", file=sys.stderr)
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

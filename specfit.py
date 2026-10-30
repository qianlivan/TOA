#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
specfit.py
==========
3D cube 谱线高斯拟合（自动 / 交互 / 自适应成分数）

依赖：numpy, scipy, matplotlib, astropy

功能
----
- 自动模式：逐像素拟合 ncomp 个高斯 + baseline
- 自适应成分数：--auto-ncomp，BIC 在 1~max_ncomp 中择优
- 交互模式：鼠标点击像素，实时查看谱线拟合
- 输出：多 HDU FITS（振幅、中心速度、FWHM、χ²、ncomp）+ PNG 参数图

用法
----
单高斯批量拟合：
    python specfit.py --input cube.fits --ncomp 1 \\
        --param-output params.fits --no-show

4 高斯固定拟合：
    python specfit.py --input cube.fits --ncomp 4 \\
        --param-output params.fits --no-show

自动成分数（BIC 在 1~6 中选）：
    python specfit.py --input cube.fits --auto-ncomp --max-ncomp 6 \\
        --snr-min 4 --nproc 4 \\
        --param-output params.fits --plot-output param_maps --no-show

交互模式：
    python specfit.py --input cube.fits --mode interactive

查看 cube：
    python specfit.py --input cube.fits --info
"""

import argparse
import os
import sys
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

try:
    from multiprocessing import Pool
    HAS_MP = True
except ImportError:
    HAS_MP = False

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
    from astropy.wcs import WCS, FITSFixedWarning
    import astropy.units as u
    warnings.simplefilter("ignore", FITSFixedWarning)
except ImportError:
    print("[错误] 需要 astropy：pip install astropy", file=sys.stderr)
    sys.exit(2)


# ==================== 数据加载 ====================
def load_cube(path, hdu_spec=None):
    if not os.path.exists(path):
        raise ValueError(f"文件不存在：{path}")
    with fits.open(path) as hdul:
        if hdu_spec is None or str(hdu_spec).strip().lower() == "auto":
            idx = None
            for i, h in enumerate(hdul):
                if h.data is not None and h.data.ndim == 3:
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"{path}: 找不到 3D cube HDU")
        else:
            try:
                idx = int(hdu_spec)
            except ValueError:
                raise ValueError(f"无效 --hdu={hdu_spec!r}")
        hdu = hdul[idx]
        if hdu.data is None:
            raise ValueError(f"{path}: HDU[{idx}] 无数据")
        if hdu.data.ndim != 3:
            raise ValueError(f"{path}: HDU[{idx}] 不是 3D")

        cube = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
        unit = header.get("BUNIT", "").strip()
        vel, vel_unit = get_velocity_axis(header, cube.shape[0])
        return dict(cube=cube, header=header, hdu_index=idx, path=path,
                    unit=unit, vel=vel, vel_unit=vel_unit)


def get_velocity_axis(header, nchan):
    try:
        w = WCS(header)
    except Exception:
        return np.arange(nchan, dtype=float), "channel"
    if w.naxis != 3:
        return np.arange(nchan, dtype=float), "channel"
    try:
        ctypes = [c.upper() for c in w.wcs.ctype]
        spec_axes = [i for i, c in enumerate(ctypes)
                     if any(k in c for k in ("FREQ", "VRAD", "VOPT", "VELO",
                                              "WAVE", "AWAV", "ENER"))]
        if not spec_axes:
            return np.arange(nchan, dtype=float), "channel"
        spec_axis = spec_axes[0]
        pix = np.arange(nchan, dtype=float)
        pix_in = [np.zeros(nchan)] * w.naxis
        pix_in[spec_axis] = pix
        world = w.all_pix2world(*pix_in, 0)
        world_spec = world[spec_axis]
        ctype = ctypes[spec_axis]
        unit = w.wcs.cunit[spec_axis]
        try:
            if "FREQ" in ctype:
                freq = world_spec * unit
                return freq.to(u.km / u.s, equivalencies=u.spectral()).value, "km/s"
            elif "WAVE" in ctype:
                wave = world_spec * unit
                return wave.to(u.km / u.s, equivalencies=u.spectral()).value, "km/s"
            elif "VELO" in ctype or "VRAD" in ctype or "VOPT" in ctype:
                return (world_spec * unit).to(u.km / u.s).value, "km/s"
        except Exception:
            return world_spec, str(unit)
    except Exception:
        pass
    return np.arange(nchan, dtype=float), "channel"


# ==================== 高斯模型 ====================
def multi_gauss(x, *params):
    n = (len(params) - 1) // 3
    y = np.full_like(x, params[-1], dtype=float)
    for i in range(n):
        a = params[3 * i]
        mu = params[3 * i + 1]
        s = max(abs(params[3 * i + 2]), 1e-10)
        y = y + a * np.exp(-0.5 * ((x - mu) / s) ** 2)
    return y


def initial_guess(vel, flux, ncomp, baseline="const"):
    vel = np.asarray(vel, dtype=float)
    flux = np.asarray(flux, dtype=float)
    dv = abs(vel[1] - vel[0]) if len(vel) > 1 else 1.0

    if baseline == "linear":
        edge = max(2, len(vel) // 10)
        left = np.nanmedian(flux[:edge])
        right = np.nanmedian(flux[-edge:])
        slope = (right - left) / (vel[-1] - vel[0]) if vel[-1] != vel[0] else 0.0
        intercept = left - slope * vel[0]
        resid = flux - (intercept + slope * vel)
    else:
        b0 = np.nanpercentile(flux, 15)
        intercept, slope = float(b0), 0.0
        resid = flux - b0

    resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.maximum(resid, 0)

    if ncomp == 1:
        idx = int(np.argmax(resid))
        i0, i1 = max(0, idx - 20), min(len(vel), idx + 21)
        v_l = vel[i0:i1]
        y_l = pos[i0:i1]
        if y_l.sum() > 0:
            mu = float(np.sum(v_l * y_l) / y_l.sum())
            sig = float(np.sqrt(np.sum(y_l * (v_l - mu) ** 2) / y_l.sum()))
            sig = max(sig, dv)
        else:
            mu = float(vel[idx])
            sig = float((vel[-1] - vel[0]) / 10)
        amps, mus, sigs = [float(pos[idx])], [mu], [sig]
    else:
        try:
            peaks, _ = find_peaks(pos, height=0)
        except Exception:
            peaks = np.array([], dtype=int)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(pos))])
        order = np.argsort(pos[peaks])[::-1]
        peaks = peaks[order[:ncomp]]
        if len(peaks) < ncomp:
            extra = np.linspace(0, len(vel) - 1,
                                ncomp - len(peaks) + 1).astype(int)
            peaks = np.concatenate([peaks, extra])[:ncomp]
        peaks = np.sort(peaks)

        amps, mus, sigs = [], [], []
        for p in peaks:
            i0, i1 = max(0, p - 15), min(len(vel), p + 16)
            v_l = vel[i0:i1]
            y_l = pos[i0:i1]
            if y_l.sum() > 0:
                mu = float(np.sum(v_l * y_l) / y_l.sum())
                sig = float(np.sqrt(np.sum(y_l * (v_l - mu) ** 2) / y_l.sum()))
                sig = max(sig, dv)
            else:
                mu = float(vel[p])
                sig = float((vel[-1] - vel[0]) / 10)
            amps.append(max(float(pos[p]), 1e-10))
            mus.append(mu)
            sigs.append(sig)

    p0 = []
    for a, m, s in zip(amps, mus, sigs):
        p0.extend([a, m, s])
    if baseline == "linear":
        p0.extend([intercept, slope])
    else:
        p0.append(intercept)
    return np.array(p0, dtype=float)


def make_bounds(vel, ncomp, baseline="const"):
    vmin = float(np.nanmin(vel))
    vmax = float(np.nanmax(vel))
    dv = abs(vel[1] - vel[0]) if len(vel) > 1 else 1.0
    lo, hi = [], []
    for _ in range(ncomp):
        lo += [-np.inf, vmin, dv * 0.1]
        hi += [np.inf, vmax, (vmax - vmin) * 2]
    if baseline == "linear":
        lo += [-np.inf, -np.inf]
        hi += [np.inf, np.inf]
    else:
        lo.append(-np.inf)
        hi.append(np.inf)
    return np.array(lo), np.array(hi)


def fit_pixel(vel, flux, ncomp=1, baseline="const", noise=None):
    """单像素固定 ncomp 拟合。返回 (popt, perr, chi2, ok)。"""
    vel = np.asarray(vel, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(flux)
    if finite.sum() < 3 * ncomp + 2:
        return None, None, np.inf, False
    v = vel[finite]
    f = flux[finite]
    try:
        p0 = initial_guess(v, f, ncomp, baseline)
        lo, hi = make_bounds(v, ncomp, baseline)
        p0 = np.clip(p0,
                     np.where(np.isfinite(lo), lo + 1e-10, -np.inf),
                     np.where(np.isfinite(hi), hi - 1e-10, np.inf))
        sigma = (np.full_like(f, noise)
                 if (noise is not None and noise > 0) else None)
        popt, pcov = curve_fit(
            multi_gauss, v, f, p0=p0,
            bounds=(lo, hi), maxfev=5000,
            sigma=sigma, absolute_sigma=False,
        )
        perr = np.sqrt(np.abs(np.diag(pcov)))
        resid = f - multi_gauss(v, *popt)
        dof = max(1, len(v) - len(p0))
        chi2 = float(np.sum(resid ** 2) / dof)
        return popt, perr, chi2, True
    except Exception:
        return None, None, np.inf, False


# ==================== ★ 新增：BIC/AIC 自动选成分数 ====================
def compute_bic_aic(f, model, k):
    """返回 (BIC, AIC)，越小越好。"""
    resid = f - model
    rss = float(np.sum(resid ** 2))
    n = len(f)
    if rss <= 0 or n <= 0 or k <= 0:
        return np.inf, np.inf
    log_term = n * np.log(max(rss, 1e-30) / n)
    bic = log_term + k * np.log(n)
    aic = log_term + 2 * k
    return float(bic), float(aic)


def fit_pixel_auto(vel, flux, max_ncomp=5, baseline="const", noise=None):
    """在 1..max_ncomp 中按 BIC 选最优。返回 (popt, perr, chi2, ok, chosen)。"""
    vel = np.asarray(vel, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(flux)
    if finite.sum() < 5:
        return None, None, np.inf, False, 1
    v = vel[finite]
    f = flux[finite]
    n = len(f)

    best = None
    best_bic = np.inf
    best_k = 1

    for ncomp in range(1, max_ncomp + 1):
        k_min = 3 * ncomp + (2 if baseline == "linear" else 1)
        if n < k_min + 3:           # 数据点太少，无法可靠拟合
            break
        popt, perr, chi2, ok = fit_pixel(v, f, ncomp, baseline, noise)
        if not ok:
            continue
        model = multi_gauss(v, *popt)
        bic, _ = compute_bic_aic(f, model, len(popt))
        if bic < best_bic:
            best_bic = bic
            best = (popt, perr, chi2, ok)
            best_k = ncomp

    if best is None:
        return None, None, np.inf, False, 1
    popt, perr, chi2, ok = best
    return popt, perr, chi2, ok, best_k


# ==================== 噪声 & 批量拟合 ====================
def estimate_noise(cube):
    nchan = cube.shape[0]
    n_edge = max(5, nchan // 10)
    edges = np.concatenate([cube[:n_edge].ravel(), cube[-n_edge:].ravel()])
    edges = edges[np.isfinite(edges)]
    if edges.size == 0:
        return 1.0
    med = np.median(edges)
    mad = np.median(np.abs(edges - med))
    return float(1.4826 * mad) if mad > 0 else float(np.std(edges))


def _fit_worker(task):
    (iy, ix, vel, flux, ncomp, baseline, noise, auto, max_ncomp) = task
    if auto:
        popt, perr, chi2, ok, chosen = fit_pixel_auto(
            vel, flux, max_ncomp, baseline, noise)
    else:
        popt, perr, chi2, ok = fit_pixel(vel, flux, ncomp, baseline, noise)
        chosen = ncomp
    return iy, ix, popt, perr, chi2, ok, chosen


def fit_cube(cube, vel, ncomp, baseline, snr_min, nproc,
             auto_ncomp=False, max_ncomp=5, verbose=True):
    nchan, ny, nx = cube.shape
    noise = estimate_noise(cube)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak_map = np.nanmax(cube, axis=0)
    snr = peak_map / max(noise, 1e-10)

    tasks = []
    for iy in range(ny):
        for ix in range(nx):
            if not np.isfinite(snr[iy, ix]) or snr[iy, ix] < snr_min:
                continue
            spec = cube[:, iy, ix]
            if not np.any(np.isfinite(spec)):
                continue
            tasks.append((iy, ix, vel, spec, ncomp, baseline, noise,
                          auto_ncomp, max_ncomp))

    if verbose:
        print(f"[信息] 有效像素 {len(tasks)}/{ny*nx} "
              f"(噪声={noise:.4g}, S/N>={snr_min})"
              + (" [BIC 自适应]" if auto_ncomp else f" [ncomp={ncomp}]"))

    ncomp_out = max_ncomp if auto_ncomp else ncomp
    npar = 3 * ncomp_out + (2 if baseline == "linear" else 1)

    params = np.full((npar, ny, nx), np.nan, dtype=np.float32)
    errors = np.full((npar, ny, nx), np.nan, dtype=np.float32)
    chi2 = np.full((ny, nx), np.nan, dtype=np.float32)
    ncomp_map = np.zeros((ny, nx), dtype=np.int16)

    results = []
    if nproc > 1 and len(tasks) > 50 and HAS_MP:
        try:
            with Pool(nproc) as pool:
                chunk = max(1, len(tasks) // (nproc * 4))
                results = pool.map(_fit_worker, tasks, chunksize=chunk)
        except Exception as e:
            if verbose:
                print(f"[警告] 多进程失败，退化串行：{e}")
            results = [_fit_worker(t) for t in tasks]
    else:
        total = len(tasks)
        for i, t in enumerate(tasks):
            if verbose and total > 20 and (i + 1) % max(1, total // 10) == 0:
                print(f"  ... {i+1}/{total}")
            results.append(_fit_worker(t))

    for iy, ix, popt, perr, c2, ok, chosen in results:
        if ok:
            m = len(popt)
            params[:m, iy, ix] = popt
            errors[:m, iy, ix] = perr
            chi2[iy, ix] = c2
            ncomp_map[iy, ix] = chosen

    keys = []
    for i in range(ncomp_out):
        keys += [f"amp{i+1}", f"v{i+1}", f"sigma{i+1}"]
    keys += ["c0"] if baseline == "const" else ["c0", "c1"]

    param_maps = {k: params[i] for i, k in enumerate(keys)}
    error_maps = {k: errors[i] for i, k in enumerate(keys)}

    if verbose and auto_ncomp:
        valid = ncomp_map[ncomp_map > 0]
        if valid.size > 0:
            hist = np.bincount(valid, minlength=max_ncomp + 1)
            print("[信息] BIC 选择的成分数分布：")
            for k in range(1, max_ncomp + 1):
                if k < len(hist):
                    print(f"   ncomp={k}: {hist[k]} 像素")

    return {
        "params": param_maps,
        "errors": error_maps,
        "chi2": chi2,
        "ncomp_map": ncomp_map,
        "ncomp": ncomp_out,
        "auto_ncomp": auto_ncomp,
        "max_ncomp": max_ncomp,
        "baseline": baseline,
        "noise": noise,
        "vel": vel,
    }


# ==================== 输出 ====================
def save_params_fits(results, header, output, verbose=True):
    hdul = fits.HDUList([fits.PrimaryHDU()])
    meta = fits.Header()
    for key in ("OBJECT", "TELESCOP", "INSTRUME", "BUNIT", "ORIGIN"):
        if key in header:
            meta[key] = header[key]
    meta["NSCOMP"] = (results["ncomp"], "max number of Gaussian components")
    meta["AUTO_N"] = (results["auto_ncomp"], "BIC auto-selection used")
    meta["MAX_NC"] = (results["max_ncomp"], "max ncomp tested in auto mode")
    meta["BASELINE"] = (results["baseline"], "baseline model")
    meta["NOISE"] = (float(results["noise"]), "estimated noise")

    for k, arr in results["params"].items():
        if np.all(~np.isfinite(arr)):
            continue  # 完全无数据的成分组不写入
        hdul.append(fits.ImageHDU(arr, name=k, header=meta.copy()))
    for k, arr in results["errors"].items():
        if np.all(~np.isfinite(arr)):
            continue
        hdul.append(fits.ImageHDU(arr, name=f"{k}_err", header=meta.copy()))
    hdul.append(fits.ImageHDU(results["chi2"], name="chi2",
                              header=meta.copy()))
    hdul.append(fits.ImageHDU(results["ncomp_map"], name="ncomp",
                              header=meta.copy()))
    hdul.writeto(output, overwrite=True)
    if verbose:
        print(f"[保存] 参数 FITS -> {output}")


def save_param_pngs(results, prefix, args, verbose=True):
    for k, arr in results["params"].items():
        if np.all(~np.isfinite(arr)):
            continue
        fig, ax = plt.subplots(figsize=(5, 4.5))
        data = arr
        label = k
        if k.startswith("sigma"):
            data = arr * 2.3548
            label = k.replace("sigma", "FWHM")
        im = ax.imshow(data, origin="lower", cmap=args.cmap,
                       interpolation="nearest")
        ax.set_title(label)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        fig.colorbar(im, ax=ax, label=label)
        out = f"{prefix}_{label}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"[保存] {out}")

    for key, name in (("chi2", "chi2_red"), ("ncomp_map", "ncomp")):
        arr = results.get(key)
        if arr is None or np.all(~np.isfinite(arr)):
            continue
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(arr, origin="lower", cmap=args.cmap,
                       interpolation="nearest")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label=name)
        out = f"{prefix}_{name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"[保存] {out}")


# ==================== 交互模式 ====================
class SpecFitViewer:
    def __init__(self, cube, vel, vel_unit, args):
        self.cube = cube
        self.vel = vel
        self.vel_unit = vel_unit
        self.args = args
        self.nchan, self.ny, self.nx = cube.shape
        self._build()

    def _build(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mom0 = np.nansum(self.cube, axis=0)
        self.mom0 = np.where(np.isfinite(self.mom0), self.mom0, 0)

        self.fig, (self.ax_img, self.ax_spec) = plt.subplots(1, 2,
                                                             figsize=(12, 5.5))
        try:
            self.fig.canvas.manager.set_window_title("SpecFit")
        except Exception:
            pass
        self.im = self.ax_img.imshow(self.mom0, origin="lower",
                                     cmap=self.args.cmap,
                                     interpolation="nearest")
        self.ax_img.set_title("Click pixel to fit")
        self.fig.colorbar(self.im, ax=self.ax_img,
                          fraction=0.046, pad=0.04, label="moment-0")
        self.ax_spec.set_xlabel(f"velocity [{self.vel_unit}]")
        self.ax_spec.set_ylabel("flux")
        self.ax_spec.set_title("(click on the left image)")
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_click(self, event):
        if event.inaxes != self.ax_img or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        ix = int(round(event.xdata))
        iy = int(round(event.ydata))
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return
        self._fit_and_plot(iy, ix)

    def _fit_and_plot(self, iy, ix):
        spec = self.cube[:, iy, ix]
        if self.args.auto_ncomp:
            popt, perr, chi2, ok, chosen = fit_pixel_auto(
                self.vel, spec, self.args.max_ncomp, self.args.baseline)
        else:
            popt, perr, chi2, ok = fit_pixel(
                self.vel, spec, self.args.ncomp, self.args.baseline)
            chosen = self.args.ncomp

        self.ax_spec.clear()
        self.ax_spec.plot(self.vel, spec, "k.", ms=2, label="data")
        if ok:
            model = multi_gauss(self.vel, *popt)
            self.ax_spec.plot(self.vel, model, "r-", lw=1.2,
                              label="total fit")
            b0 = popt[-1]
            self.ax_spec.axhline(b0, color="gray", ls=":", lw=0.8,
                                 label=f"baseline={b0:.3g}")
            for i in range(chosen):
                a, mu, s = popt[3 * i:3 * i + 3]
                comp = b0 + a * np.exp(-0.5 * ((self.vel - mu) / s) ** 2)
                self.ax_spec.plot(self.vel, comp, "--", lw=0.8, alpha=0.7,
                                  label=f"c{i+1}: μ={mu:.2f}, "
                                        f"FWHM={2.355*abs(s):.2f}")
            tag = f"ncomp={chosen}" if self.args.auto_ncomp else ""
            self.ax_spec.set_title(
                f"pixel ({ix},{iy})   χ²_red={chi2:.3g}   {tag}")
        else:
            self.ax_spec.set_title(f"pixel ({ix},{iy})   fit failed")
        self.ax_spec.set_xlabel(f"velocity [{self.vel_unit}]")
        self.ax_spec.set_ylabel("flux")
        self.ax_spec.legend(fontsize=7, loc="best")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key == "s":
            self.fig.savefig("specfit_pixel.png",
                             dpi=self.args.dpi, bbox_inches="tight")
            print("[保存] specfit_pixel.png")

    def show(self):
        try:
            plt.show()
        except Exception as e:
            print(f"[警告] 无法显示：{e}", file=sys.stderr)


# ==================== CLI ====================
def _split_csv(s):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser(
        description="3D cube 谱线高斯拟合（自动 / 交互 / BIC 自适应成分数）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python specfit.py --input cube.fits --ncomp 2 --param-output p.fits --no-show
  python specfit.py --input cube.fits --ncomp 4 --snr-min 4 --nproc 4 --no-show
  python specfit.py --input cube.fits --auto-ncomp --max-ncomp 6 --no-show
  python specfit.py --input cube.fits --mode interactive
  python specfit.py --input cube.fits --info
""")

    src = p.add_argument_group("数据源")
    src.add_argument("--input", required=True, help="3D cube FITS 路径")
    src.add_argument("--hdu", default="auto",
                     help="HDU 索引（默认 auto 找 3D）")
    src.add_argument("--info", action="store_true",
                     help="打印 cube 信息后退出")

    fit = p.add_argument_group("拟合参数")
    fit.add_argument("--ncomp", type=int, default=1,
                     help="固定的高斯成分数（1..8，auto 模式下忽略）")
    fit.add_argument("--auto-ncomp", action="store_true",
                     help="BIC 在 1..max-ncomp 中自动选择成分数")
    fit.add_argument("--max-ncomp", type=int, default=5,
                     help="auto 模式下最大尝试的成分数（默认 5）")
    fit.add_argument("--baseline", choices=["const", "linear"], default="const",
                     help="基线模型")
    fit.add_argument("--velocity-range", default=None,
                     help="拟合速度范围，格式 vmin,vmax")
    fit.add_argument("--snr-min", type=float, default=3.0,
                     help="峰值 S/N 筛选阈值（默认 3）")
    fit.add_argument("--nproc", type=int, default=1,
                     help="并行进程数（默认 1）")

    mode = p.add_argument_group("模式")
    mode.add_argument("--mode", choices=["auto", "interactive"],
                      default="auto",
                      help="auto=批量拟合；interactive=交互查看")

    out = p.add_argument_group("输出")
    out.add_argument("--param-output", default="specfit_params.fits",
                     help="参数 FITS 输出路径")
    out.add_argument("--plot-output", default=None,
                     help="参数图 PNG 前缀（留空=不输出）")
    out.add_argument("--cmap", default="viridis")
    out.add_argument("--dpi", type=int, default=150)
    out.add_argument("--no-show", action="store_true",
                     help="不弹窗（自动模式）")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    # 参数合理性
    if not args.auto_ncomp and args.ncomp < 1:
        print("[错误] --ncomp 必须 >= 1", file=sys.stderr)
        return 1
    if args.auto_ncomp and args.max_ncomp < 1:
        print("[错误] --max-ncomp 必须 >= 1", file=sys.stderr)
        return 1

    try:
        data = load_cube(args.input, args.hdu)
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    cube = data["cube"]
    vel_full = data["vel"]
    nchan, ny, nx = cube.shape

    if args.info:
        print(f"文件：{args.input}")
        print(f"HDU：{data['hdu_index']}")
        print(f"shape：{cube.shape} (nchan={nchan}, ny={ny}, nx={nx})")
        print(f"速度轴：{vel_full[0]:.4g} → {vel_full[-1]:.4g} "
              f"{data['vel_unit']}  (共 {nchan} 通道)")
        print(f"BUNIT：{data['unit'] or '(未定义)'}")
        return 0

    vel = vel_full
    if args.velocity_range:
        parts = _split_csv(args.velocity_range)
        if len(parts) != 2:
            print("[错误] --velocity-range 需要 'vmin,vmax'", file=sys.stderr)
            return 1
        vmin, vmax = float(parts[0]), float(parts[1])
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        keep = (vel >= vmin) & (vel <= vmax)
        if not keep.any():
            print(f"[错误] 速度范围 {vmin}~{vmax} 内无通道", file=sys.stderr)
            return 1
        cube = cube[keep]
        vel = vel[keep]
        print(f"[信息] 裁剪到 {keep.sum()} 通道 "
              f"({vmin:.2f} ~ {vmax:.2f} {data['vel_unit']})")

    if args.mode == "interactive":
        SpecFitViewer(cube, vel, data["vel_unit"], args).show()
        return 0

    print(f"[信息] 拟合 baseline={args.baseline}, snr_min={args.snr_min}"
          + (f", auto_ncomp (max={args.max_ncomp})"
             if args.auto_ncomp else f", ncomp={args.ncomp}"))

    results = fit_cube(
        cube, vel,
        ncomp=args.ncomp,
        baseline=args.baseline,
        snr_min=args.snr_min,
        nproc=args.nproc,
        auto_ncomp=args.auto_ncomp,
        max_ncomp=args.max_ncomp,
    )

    if args.param_output:
        save_params_fits(results, data["header"], args.param_output)
    if args.plot_output:
        save_param_pngs(results, args.plot_output, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())

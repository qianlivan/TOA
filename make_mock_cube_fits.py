#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_mock_cube_fits.py
======================
生成模拟 3D 谱线 datacube（银道 CAR 投影 + 速度轴）

每个空间像素的谱线 = 若干高斯成分（1..ncomp-max）
                   + 常数基线 + 高斯白噪声
速度场可模拟简单旋转盘（v = v_disk · sin(l)）

依赖：numpy, astropy

用法
----
默认 64 通道 × 180 × 360，2 个高斯成分（含消隐），速度轴为 VRAD：
    python make_mock_cube_fits.py

较大 cube，频率轴，含旋转速度场：
    python make_mock_cube_fits.py --shape 128 360 720 \\
        --spec-type freq --freq-center 1.420405751e9 \\
        --bandwidth 1e7 --v-disk 150

同时输出 Moment-0 2D 和真值 FITS（方便与 specfit 输出对比）：
    python make_mock_cube_fits.py \\
        --output mock_cube.fits \\
        --moment0 mock_cube_mom0.fits \\
        --save-truth mock_cube_truth.fits

只生成纯单高斯（用于测试 ncomp=1）：
    python make_mock_cube_fits.py --ncomp-max 1 --output cube1.fits
"""

import argparse
import os
import sys

import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    import warnings
    warnings.simplefilter("ignore", FITSFixedWarning)
except ImportError:
    print("[错误] 需要 astropy：pip install astropy", file=sys.stderr)
    sys.exit(2)


C_KMS = 299792.458  # km/s


# ==================== WCS 构造 ====================
def make_cube_wcs(nchan, ny, nx, spec_type="vrad",
                  vrange=(-300.0, 300.0),
                  freq_center=1.420405751e9, bandwidth=1e7):
    """返回 3D WCS 对象，轴顺序 = (spec, lat, lon) → 转成 FITS 头时是 (lon, lat, spec)。

    注意：FITS 里 NAXIS1 = lon（x），NAXIS2 = lat（y），NAXIS3 = spec。
    而 numpy 数组是 (nchan, ny, nx)，对应 FITS 的 (NAXIS3, NAXIS2, NAXIS1)。
    """
    w = WCS(naxis=3)

    # 空间轴（GLON-CAR / GLAT-CAR）
    w.wcs.ctype[0] = "GLON-CAR"
    w.wcs.ctype[1] = "GLAT-CAR"
    w.wcs.cunit[0] = "deg"
    w.wcs.cunit[1] = "deg"
    w.wcs.crval[0] = 0.0
    w.wcs.crval[1] = 0.0
    w.wcs.crpix[0] = (nx + 1) / 2.0
    w.wcs.crpix[1] = (ny + 1) / 2.0
    w.wcs.cdelt[0] = -360.0 / nx
    w.wcs.cdelt[1] = 180.0 / ny

    # 频谱轴
    if spec_type == "vrad":
        vmin, vmax = vrange
        w.wcs.ctype[2] = "VRAD"
        w.wcs.cunit[2] = "m/s"
        w.wcs.crval[2] = vmin * 1e3          # 第 0 像素 = vmin
        w.wcs.crpix[2] = 1.0
        w.wcs.cdelt[2] = (vmax - vmin) / max(nchan - 1, 1) * 1e3
    else:  # freq
        dfreq = bandwidth / nchan
        w.wcs.ctype[2] = "FREQ"
        w.wcs.cunit[2] = "Hz"
        w.wcs.crval[2] = freq_center
        w.wcs.crpix[2] = (nchan + 1) / 2.0
        w.wcs.cdelt[2] = dfreq

    return w


# ==================== 速度轴 ====================
def get_velocity_axis(nchan, spec_type, vrange, freq_center, bandwidth):
    """返回 km/s 速度数组（与 cube 的 channel 一一对应）。"""
    if spec_type == "vrad":
        vmin, vmax = vrange
        return np.linspace(vmin, vmax, nchan)
    # freq：f = f0 + (i - (n-1)/2) * dfreq
    dfreq = bandwidth / nchan
    i = np.arange(nchan) - (nchan - 1) / 2.0
    freq = freq_center + i * dfreq
    # 光速近似：v = c · (f0 - f) / f0
    return C_KMS * (freq_center - freq) / freq_center


# ==================== 生成 ====================
def generate_mock_cube(shape=(64, 180, 360),
                       vrange=(-300.0, 300.0),
                       spec_type="vrad",
                       freq_center=1.420405751e9,
                       bandwidth=1e7,
                       ncomp_max=2,
                       v_disk=120.0,
                       v_scatter=15.0,
                       sigma_range=(8.0, 25.0),
                       amp_range=(0.3, 1.0),
                       baseline=0.1,
                       noise=0.02,
                       l_cutoff=None,
                       seed=42):
    """返回 (cube[nchan,ny,nx], vel[nchan], truth dict, wcs)。"""
    nchan, ny, nx = shape
    rng = np.random.default_rng(seed)

    wcs = make_cube_wcs(nchan, ny, nx, spec_type, vrange,
                        freq_center, bandwidth)
    vel = get_velocity_axis(nchan, spec_type, vrange, freq_center, bandwidth)

    # ---- 空间像素中心 (lon, lat) ----
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    lon, lat = wcs.celestial.all_pix2world(xx, yy, 0)

    # ---- 每像素高斯成分数 ----
    ncomp_map = rng.integers(1, ncomp_max + 1, size=(ny, nx)).astype(np.int16)

    # ---- 每像素每成分的参数图 ----
    amps = np.zeros((ncomp_max, ny, nx), dtype=np.float32)
    mus = np.zeros((ncomp_max, ny, nx), dtype=np.float32)
    sigs = np.zeros((ncomp_max, ny, nx), dtype=np.float32)

    # 旋转盘速度场 + 随机散布
    v_field = v_disk * np.sin(np.radians(lon)) + rng.normal(
        0, v_scatter, size=(ny, nx))

    for k in range(ncomp_max):
        # 对该成分"存在"的像素（ncomp_map > k）给参数
        active = ncomp_map > k
        # 第二个及以后成分：在速度场上偏开，制造多峰感
        if k == 0:
            mu = v_field.copy()
        else:
            mu = v_field + rng.normal(0, 30, size=(ny, nx)) * k

        sig = rng.uniform(*sigma_range, size=(ny, nx))
        amp = rng.uniform(*amp_range, size=(ny, nx))

        amps[k] = np.where(active, amp, 0.0)
        mus[k] = np.where(active, mu, 0.0)
        sigs[k] = np.where(active, sig, 1.0)

    # ---- 谱线：广播生成整个 cube ----
    cube = np.full((nchan, ny, nx), baseline, dtype=np.float32)
    v3 = vel[:, None, None]
    for k in range(ncomp_max):
        a3 = amps[k][None, :, :]
        m3 = mus[k][None, :, :]
        s3 = np.maximum(sigs[k][None, :, :], 1e-6)
        cube += (a3 * np.exp(-0.5 * ((v3 - m3) / s3) ** 2)).astype(np.float32)

    # ---- 噪声 ----
    if noise > 0:
        cube += rng.normal(0, noise, size=cube.shape).astype(np.float32)

    # ---- 可选：只保留某些经度范围（用于小视场测试） ----
    if l_cutoff is not None:
        lmin, lmax = l_cutoff
        keep = (lon >= lmin) & (lon <= lmax)
        cube[:, ~keep] = baseline
        amps[:, ~keep] = 0.0
        ncomp_map[~keep] = 1

    truth = dict(
        ncomp=ncomp_map,
        amps=amps, mus=mus, sigmas=sigs,
        vel=vel, baseline=baseline, noise=noise,
    )
    return cube, vel, truth, wcs


# ==================== 输出 ====================
def save_cube(path, cube, wcs, truth, verbose=True):
    header = wcs.to_header()
    header["BUNIT"] = ("K", "mock brightness temperature")
    header["OBJECT"] = "MOCK CUBE"
    header["ORIGIN"] = "make_mock_cube_fits.py"
    header["NSCOMP"] = (truth["ncomp"].max(), "max Gaussian components per pixel")
    header["BASELINE"] = (truth["baseline"], "constant baseline")
    header["NOISE"] = (truth["noise"], "noise RMS")

    # FITS 轴顺序 (nchan, ny, nx) 对应 NAXIS3, NAXIS2, NAXIS1，正好匹配
    fits.writeto(path, cube.astype(np.float32), header, overwrite=True)
    if verbose:
        print(f"[保存] cube -> {path}  shape={cube.shape}")


def save_moment0(path, cube, wcs, verbose=True):
    with np.errstate(invalid="ignore"):
        mom0 = np.nansum(cube, axis=0)
    header = wcs.celestial.to_header()
    header["BUNIT"] = ("K*channel", "moment-0")
    header["OBJECT"] = "MOCK CUBE MOMENT-0"
    fits.writeto(path, mom0.astype(np.float32), header, overwrite=True)
    if verbose:
        print(f"[保存] moment-0 -> {path}  shape={mom0.shape}")


def save_truth(path, truth, wcs, verbose=True):
    """真值 FITS：每成分一个 amp / v / sigma HDU + ncomp HDU。"""
    hdr = wcs.celestial.to_header()
    hdr["BASELINE"] = (truth["baseline"], "true constant baseline")
    hdr["NOISE"] = (truth["noise"], "true noise RMS")

    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])
    ncomp_max = truth["amps"].shape[0]
    for k in range(ncomp_max):
        hdul.append(fits.ImageHDU(
            truth["amps"][k].astype(np.float32),
            name=f"AMP{k+1}", header=hdr.copy()))
        hdul.append(fits.ImageHDU(
            truth["mus"][k].astype(np.float32),
            name=f"V{k+1}", header=hdr.copy()))
        hdul.append(fits.ImageHDU(
            truth["sigmas"][k].astype(np.float32),
            name=f"SIGMA{k+1}", header=hdr.copy()))
    hdul.append(fits.ImageHDU(truth["ncomp"].astype(np.int16),
                              name="NCOMP", header=hdr.copy()))
    hdul.writeto(path, overwrite=True)
    if verbose:
        print(f"[保存] truth -> {path}  ({ncomp_max} 个可能成分)")


# ==================== CLI ====================
def _split_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser(
        description="生成模拟 3D 谱线 datacube（银道 CAR 投影 + 速度轴）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python make_mock_cube_fits.py
  python make_mock_cube_fits.py --shape 128 360 720 --v-disk 150
  python make_mock_cube_fits.py --spec-type freq --bandwidth 1e7
  python make_mock_cube_fits.py --ncomp-max 1 --output cube1.fits
  python make_mock_cube_fits.py --save-truth truth.fits
""")

    shp = p.add_argument_group("尺寸与轴")
    shp.add_argument("--shape", nargs=3, type=int,
                     default=[64, 180, 360], metavar=("NCHAN", "NY", "NX"),
                     help="维度，默认 64 180 360")
    shp.add_argument("--vrange", nargs=2, type=float, default=[-300.0, 300.0],
                     metavar=("VMIN", "VMAX"),
                     help="速度范围（km/s），默认 -300 300")
    shp.add_argument("--spec-type", choices=["vrad", "freq"], default="vrad",
                     help="频谱轴类型（默认 vrad）")
    shp.add_argument("--freq-center", type=float, default=1.420405751e9,
                     help="频率轴中心（Hz），仅 spec-type=freq")
    shp.add_argument("--bandwidth", type=float, default=1e7,
                     help="带宽（Hz），仅 spec-type=freq")

    sig = p.add_argument_group("信号")
    sig.add_argument("--ncomp-max", type=int, default=2,
                     help="每像素最多高斯成分数（默认 2）")
    sig.add_argument("--v-disk", type=float, default=120.0,
                     help="旋转盘速度幅度（km/s），默认 120")
    sig.add_argument("--v-scatter", type=float, default=15.0,
                     help="速度场随机散布（km/s）")
    sig.add_argument("--sigma-range", nargs=2, type=float,
                     default=[8.0, 25.0],
                     help="高斯 sigma 范围（km/s），默认 8 25")
    sig.add_argument("--amp-range", nargs=2, type=float,
                     default=[0.3, 1.0],
                     help="高斯峰值强度范围")
    sig.add_argument("--baseline", type=float, default=0.1,
                     help="常数基线（默认 0.1）")
    sig.add_argument("--noise", type=float, default=0.02,
                     help="噪声 RMS（0 = 无噪声）")
    sig.add_argument("--l-cutoff", default=None,
                     help="只保留经度范围，格式 lmin,lmax")

    out = p.add_argument_group("输出")
    out.add_argument("--output", default="mock_cube.fits",
                     help="cube FITS 路径")
    out.add_argument("--moment0", default=None,
                     help="Moment-0 2D FITS 路径（可选）")
    out.add_argument("--save-truth", default=None,
                     help="真值参数 FITS 路径（可选）")
    out.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    l_cutoff = None
    if args.l_cutoff:
        parts = _split_csv(args.l_cutoff)
        if len(parts) != 2:
            print("[错误] --l-cutoff 需要 'lmin,lmax'", file=sys.stderr)
            return 1
        l_cutoff = (float(parts[0]), float(parts[1]))

    print(f"[信息] 生成 {args.shape[0]}×{args.shape[1]}×{args.shape[2]} "
          f"(nchan×ny×nx)  spec-type={args.spec_type}")

    try:
        cube, vel, truth, wcs = generate_mock_cube(
            shape=tuple(args.shape),
            vrange=tuple(args.vrange),
            spec_type=args.spec_type,
            freq_center=args.freq_center,
            bandwidth=args.bandwidth,
            ncomp_max=args.ncomp_max,
            v_disk=args.v_disk,
            v_scatter=args.v_scatter,
            sigma_range=tuple(args.sigma_range),
            amp_range=tuple(args.amp_range),
            baseline=args.baseline,
            noise=args.noise,
            l_cutoff=l_cutoff,
            seed=args.seed,
        )
    except Exception as e:
        print(f"[错误] 生成失败：{e}", file=sys.stderr)
        return 1

    print(f"[信息] 速度轴：{vel[0]:.2f} → {vel[-1]:.2f} km/s  "
          f"(c={len(vel)})")
    print(f"[信息] ncomp 分布：", end="")
    for k in range(1, args.ncomp_max + 1):
        cnt = int((truth["ncomp"] == k).sum())
        print(f" {k}->{cnt}", end="")
    print()
    print(f"[信息] 数据均值 {cube.mean():.3f}, 噪声 RMS≈{truth['noise']}")

    save_cube(args.output, cube, wcs, truth)

    if args.moment0:
        save_moment0(args.moment0, cube, wcs)

    if args.save_truth:
        save_truth(args.save_truth, truth, wcs)

    print("[完成]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_mock_sky_fits.py
=====================
生成模拟全天 FITS 天图（银道坐标 CAR 投影）

特征：
    - 银河盘面：沿 b=0 的高斯带状结构，l=0 方向增亮
    - 点源：随机高斯轮廓
    - 弥散结构：大尺度平滑云团
    - 高斯噪声
    - 标准 WCS 头（GLON-CAR / GLAT-CAR），可直接用 skymap_fits.py 读取

依赖：numpy, astropy

用法：
    python make_mock_sky_fits.py                          # 默认 720x360
    python make_mock_sky_fits.py --shape 1440 720 --n-sources 500
    python make_mock_sky_fits.py --output mock_hi.fits --noise 0.05
    python make_mock_sky_fits.py --n-sources 0 --noise 0    # 纯银河盘面
"""

import argparse
import sys
import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
except ImportError:
    print("[错误] 需要 astropy：pip install astropy", file=sys.stderr)
    sys.exit(2)


# ==================== 基本组件 ====================
def galactic_disk(shape, l_center=0.0, scale_height=8.0,
                  amp=10.0, l_boost=1.0):
    """银河盘面：沿银道面的高斯带状结构，l=0 附近增亮。"""
    ny, nx = shape
    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90, 90, ny)
    L, B = np.meshgrid(lon, lat)

    # 纬向高斯带
    disk = amp * np.exp(-0.5 * (B / scale_height) ** 2)

    # 经向调制：银心（l=0）附近更亮，反银心方向略暗
    lon_mod = 1.0 + l_boost * np.cos(np.radians(L))
    return disk * lon_mod


def smooth_clouds(shape, n_clouds=8, amp=3.0, size_range=(10, 40), seed=1):
    """大尺度平滑云团（随机二维高斯）。"""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90, 90, ny)
    L, B = np.meshgrid(lon, lat)

    img = np.zeros(shape, dtype=float)
    for _ in range(n_clouds):
        l0 = rng.uniform(-180, 180)
        b0 = rng.uniform(-60, 60)
        s_l = rng.uniform(*size_range)
        s_b = rng.uniform(*size_range)
        a = amp * rng.uniform(0.5, 1.5)
        img += a * np.exp(
            -0.5 * ((L - l0) / s_l) ** 2
            - 0.5 * ((B - b0) / s_b) ** 2
        )
    return img


def point_sources(shape, n_sources=200, amp=5.0, sigma_range=(0.1, 0.8),
                  seed=2):
    """随机点源（高斯轮廓）。"""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90, 90, ny)
    L, B = np.meshgrid(lon, lat)

    img = np.zeros(shape, dtype=float)
    for _ in range(n_sources):
        l0 = rng.uniform(-180, 180)
        b0 = rng.uniform(-90, 90)
        s = rng.uniform(*sigma_range)
        a = amp * rng.uniform(0.2, 3.0)
        img += a * np.exp(
            -0.5 * ((L - l0) / s) ** 2
            - 0.5 * ((B - b0) / s) ** 2
        )
    return img


# ==================== WCS 头（银道 CAR 投影） ====================
def make_galactic_car_wcs(shape):
    """返回与 HI4PI CAR 投影兼容的 WCS 对象。"""
    ny, nx = shape
    w = WCS(naxis=2)
    w.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cdelt = [-360.0 / nx, 180.0 / ny]   # CAR 投影：经度通常递减
    return w


# ==================== 主生成函数 ====================
def make_mock_fits(shape=(360, 720), n_sources=300, noise=0.05,
                   disk_amp=10.0, cloud_amp=3.0, source_amp=8.0,
                   seed=42, output="mock_sky.fits"):
    ny, nx = shape
    rng = np.random.default_rng(seed)

    print(f"生成模拟天图：shape = {shape[1]}×{shape[0]} (nx×ny)")

    # 1. 银河盘面
    img = galactic_disk((ny, nx), amp=disk_amp, scale_height=8.0,
                        l_boost=1.0)

    # 2. 弥散云团
    img += smooth_clouds((ny, nx), n_clouds=8, amp=cloud_amp,
                         size_range=(10, 40), seed=seed + 1)

    # 3. 点源
    if n_sources > 0:
        img += point_sources((ny, nx), n_sources=n_sources,
                             amp=source_amp, sigma_range=(0.1, 0.8),
                             seed=seed + 2)

    # 4. 噪声
    if noise > 0:
        img += rng.normal(0, noise, (ny, nx))

    # 5. 截断负值（柱密度不能为负）
    img = np.clip(img, 0, None)

    # 6. WCS 头
    wcs = make_galactic_car_wcs((ny, nx))
    header = wcs.to_header()
    header["BUNIT"] = ("cm^-2", "Mock HI column density")
    header["OBJECT"] = "MOCK SKY"
    header["ORIGIN"] = "make_mock_sky_fits.py"

    # 7. 写入
    fits.writeto(output, img.astype(np.float32), header, overwrite=True)
    print(f"已写入 {output}  "
          f"(min={img.min():.3f}, max={img.max():.3f}, "
          f"mean={img.mean():.3f})")
    return img


# ==================== CLI ====================
def build_parser():
    p = argparse.ArgumentParser(
        description="生成模拟全天 FITS 天图（银道 CAR 投影）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python make_mock_sky_fits.py
  python make_mock_sky_fits.py --shape 1440 720 --n-sources 500
  python make_mock_sky_fits.py --output mock_hi.fits --noise 0.02
  python make_mock_sky_fits.py --n-sources 0 --noise 0   # 纯银河盘面
""")
    p.add_argument("--shape", nargs=2, type=int, default=[360, 720],
                   metavar=("NY", "NX"),
                   help="图像尺寸 (NY NX)，默认 360 720")
    p.add_argument("--n-sources", type=int, default=300,
                   help="点源数量（0 = 不加点源）")
    p.add_argument("--noise", type=float, default=0.05,
                   help="高斯噪声 RMS（0 = 不加噪声）")
    p.add_argument("--disk-amp", type=float, default=10.0,
                   help="银河盘面峰值幅度")
    p.add_argument("--cloud-amp", type=float, default=3.0,
                   help="弥散云团峰值幅度")
    p.add_argument("--source-amp", type=float, default=8.0,
                   help="点源基准幅度")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="mock_sky.fits")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    try:
        make_mock_fits(
            shape=tuple(args.shape),
            n_sources=args.n_sources,
            noise=args.noise,
            disk_amp=args.disk_amp,
            cloud_amp=args.cloud_amp,
            source_amp=args.source_amp,
            seed=args.seed,
            output=args.output,
        )
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

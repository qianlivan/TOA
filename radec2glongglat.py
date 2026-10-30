#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radec2glongglat.py
==================
赤道坐标 ↔ 银道坐标转换（J2000 历元）

依赖：仅标准库。

用法
----
正向（赤道 → 银道）：
    python radec2glongglat.py --ra 08h35m20.61125s --dec=-45d10m34.875s
    python radec2glongglat.py --ra 128.8359 --dec -45.1764 --decimal-in
    python radec2glongglat.py --ra 08:35:20.61 --dec=-45:10:34.9

反向（银道 → 赤道）：
    python radec2glongglat.py --glon 262.9 --glat -4.0 --reverse

仅输出十进制度（默认同时打印 hms/dms）：
    ... --decimal-out

参数
----
--ra          赤经。默认可自动识别 hms（如 08h35m20.6s）或十进制度
--dec         赤纬。默认可自动识别 dms（如 -45d10m34.9s）或十进制度
--glon        银经 l（度），反向转换时使用
--glat        银纬 b（度），反向转换时使用
--decimal-in  强制把 --ra / --dec 按十进制度解析
--decimal-out 仅输出十进制度
--reverse     反向：银道 → 赤道

坐标约定
--------
旋转矩阵采用 J2000 标准值（与 astropy / Hipparcos 一致）：
    A_eq→gal =
        [-0.0548755604, -0.8734370902, -0.4838350155]
        [ 0.4941094279, -0.4448296300,  0.7469822445]
        [-0.8676661490, -0.1980763734,  0.4559837762]

对应银河北极：
    α_NGP = 192.85948°,  δ_NGP = +27.12825°（J2000）
银经零点（银心方向）：
    l=0, b=0  ↔  α ≈ 266.40499°,  δ ≈ -28.93617°
"""

import argparse
import math
import re
import sys

DEG = math.pi / 180.0

# J2000 赤道 → 银道 旋转矩阵（正交，其转置即银道 → 赤道）
A_EQ2GAL = (
    (-0.0548755604, -0.8734370902, -0.4838350155),
    ( 0.4941094279, -0.4448296300,  0.7469822445),
    (-0.8676661490, -0.1980763734,  0.4559837762),
)


# ---------------------------------------------------------------------------
# 角度解析
# ---------------------------------------------------------------------------
def _parse_sexagesimal(s, is_ra):
    """把 hms / dms 字符串解析成十进制度。

    is_ra=True  → 时:分:秒，×15 得度
    is_ra=False → 度:分:秒
    """
    s = s.strip()
    if not s:
        raise ValueError("空字符串")

    sign = 1.0
    if s[0] in "+-":
        if s[0] == "-":
            sign = -1.0
        s = s[1:]

    tokens = [t for t in re.split(r"[hdmsHDMS°'\"：:\s]+", s) if t]
    if not tokens:
        raise ValueError(f"无法解析角度：{s!r}")

    try:
        vals = [float(t) for t in tokens]
    except ValueError:
        raise ValueError(f"无法解析角度：{s!r}")

    while len(vals) < 3:
        vals.append(0.0)
    h, m, sec = vals[0], vals[1], vals[2]

    if is_ra:
        if h > 24.0:
            raise ValueError(
                f"RA 的 sexagesimal 形式中小时数 {h} > 24，"
                f"若输入为十进制度请加 --decimal-in"
            )
        deg = (h + m / 60.0 + sec / 3600.0) * 15.0
    else:
        if h > 90.0:
            raise ValueError(f"赤纬的度数 {h} 超出 ±90°")
        deg = h + m / 60.0 + sec / 3600.0
    return sign * deg


def parse_angle(s, is_ra, decimal_in=False):
    """解析角度字符串为十进制度。

    decimal_in=True 时强制按十进制度解析；
    否则自动识别：含 h/d/m/s/°/'/"/:/ 空格 等分隔符 → sexagesimal，
    否则按十进制度处理。
    """
    s = s.strip()
    if not s:
        raise ValueError("空字符串")

    if decimal_in:
        return float(s)

    has_sep = bool(re.search(r"[hdmsHDMS°'\"：:]", s))
    multi_token = len(s.split()) > 1
    if has_sep or multi_token:
        return _parse_sexagesimal(s, is_ra)
    return float(s)


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------
def to_hms(deg):
    """度 → 'HHhMMmSS.SSSs'（用于 RA）。"""
    total_sec = (deg % 360.0) / 15.0 * 3600.0
    if total_sec >= 86400.0:      # 浮点取整到 24h
        total_sec = 0.0
    h = int(total_sec // 3600)
    rem = total_sec - h * 3600.0
    m = int(rem // 60)
    sec = rem - m * 60.0
    return f"{h:02d}h{m:02d}m{sec:06.3f}s"


def to_dms(deg):
    """度 → '±DDdMMmSS.SSSs'（用于 Dec / l / b）。"""
    sign = "-" if deg < 0 else "+"
    deg_abs = abs(deg)
    d = int(deg_abs)
    rem = (deg_abs - d) * 60.0
    m = int(rem)
    sec = (rem - m) * 60.0
    return f"{sign}{d:02d}d{m:02d}m{sec:06.3f}s"


# ---------------------------------------------------------------------------
# 坐标转换
# ---------------------------------------------------------------------------
def eq2gal(ra_deg, dec_deg):
    """赤道 (α, δ) → 银道 (l, b)，单位度。"""
    ra = ra_deg * DEG
    dec = dec_deg * DEG
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)

    xg = A_EQ2GAL[0][0] * x + A_EQ2GAL[0][1] * y + A_EQ2GAL[0][2] * z
    yg = A_EQ2GAL[1][0] * x + A_EQ2GAL[1][1] * y + A_EQ2GAL[1][2] * z
    zg = A_EQ2GAL[2][0] * x + A_EQ2GAL[2][1] * y + A_EQ2GAL[2][2] * z

    l = math.degrees(math.atan2(yg, xg)) % 360.0
    b = math.degrees(math.asin(max(-1.0, min(1.0, zg))))
    return l, b


def gal2eq(l_deg, b_deg):
    """银道 (l, b) → 赤道 (α, δ)，单位度。"""
    l = l_deg * DEG
    b = b_deg * DEG
    xg = math.cos(b) * math.cos(l)
    yg = math.cos(b) * math.sin(l)
    zg = math.sin(b)

    # 转置矩阵（旋转矩阵正交）
    x = A_EQ2GAL[0][0] * xg + A_EQ2GAL[1][0] * yg + A_EQ2GAL[2][0] * zg
    y = A_EQ2GAL[0][1] * xg + A_EQ2GAL[1][1] * yg + A_EQ2GAL[2][1] * zg
    z = A_EQ2GAL[0][2] * xg + A_EQ2GAL[1][2] * yg + A_EQ2GAL[2][2] * zg

    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return ra, dec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# 这些选项的取值可能是负数（以 '-' 开头），argparse 默认会误判为选项，
# 这里把它们统一改写成 --opt=value 形式。
_NEG_VALUE_OPTS = {"--dec", "--ra", "--glat", "--glon"}


def _normalize_argv(argv):
    out = []
    i = 0
    n = len(argv)
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
        description="赤道坐标 ↔ 银道坐标转换（J2000 历元）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python radec2glongglat.py --ra 08h35m20.61125s --dec=-45d10m34.875s\n"
            "  python radec2glongglat.py --ra 128.8359 --dec -45.1764 --decimal-in\n"
            "  python radec2glongglat.py --glon 262.9 --glat -4.0 --reverse\n"
        ),
    )
    p.add_argument("--ra", default=None,
                   help="赤经（hms，如 08h35m20.6s；或十进制度）")
    p.add_argument("--dec", default=None,
                   help="赤纬（dms，如 -45d10m34.9s；或十进制度）")
    p.add_argument("--glon", default=None,
                   help="银经 l（度），反向时使用")
    p.add_argument("--glat", default=None,
                   help="银纬 b（度），反向时使用")
    p.add_argument("--decimal-in", action="store_true",
                   help="强制 --ra/--dec 按十进制度解析")
    p.add_argument("--decimal-out", action="store_true",
                   help="仅输出十进制度（默认同时打印 hms/dms）")
    p.add_argument("--reverse", action="store_true",
                   help="反向：银道 → 赤道")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.reverse:
            if args.glon is None or args.glat is None:
                parser.error("反向转换需要同时提供 --glon 和 --glat")
            l = float(args.glon)
            b = float(args.glat)
            ra, dec = gal2eq(l, b)

            print("输入银道坐标：")
            print(f"  l = {l:.8f} deg   b = {b:.8f} deg")
            print("输出赤道坐标：")
            if not args.decimal_out:
                print(f"  RA  = {to_hms(ra)}   ({ra:.8f} deg)")
                print(f"  Dec = {to_dms(dec)}  ({dec:.8f} deg)")
            else:
                print(f"  RA  = {ra:.8f} deg")
                print(f"  Dec = {dec:.8f} deg")
        else:
            if args.ra is None or args.dec is None:
                parser.error("正向转换需要同时提供 --ra 和 --dec")
            ra = parse_angle(args.ra, is_ra=True, decimal_in=args.decimal_in)
            dec = parse_angle(args.dec, is_ra=False, decimal_in=args.decimal_in)
            l, b = eq2gal(ra, dec)

            print("输入赤道坐标：")
            if not args.decimal_out:
                print(f"  RA  = {args.ra}   ({ra:.8f} deg)")
                print(f"  Dec = {args.dec}  ({dec:.8f} deg)")
            else:
                print(f"  RA  = {ra:.8f} deg")
                print(f"  Dec = {dec:.8f} deg")
            print("输出银道坐标：")
            if not args.decimal_out:
                print(f"  l = {l:.8f} deg   ({to_dms(l)})")
                print(f"  b = {b:.8f} deg   ({to_dms(b)})")
            else:
                print(f"  l = {l:.8f} deg")
                print(f"  b = {b:.8f} deg")
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

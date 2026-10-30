#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dist_dm_convert.py
==================
距离 <-> 色散量 (DM) 转换

支持的模型
----------
fixed    固定电子密度：DM = ne * d
ne2001   银河系 NE2001 模型（需要 pygedm + --l --b）
ymw16    银河系 YMW16 模型（需要 pygedm + --l --b）

IGM（宇宙学）贡献
----------------
--cosmic + --z：按 Inoue 2004 公式计算

用法
----
1 kpc 银河内 DM（固定 ne=0.03）：
    python dist_dm_convert.py --distance 1.0 --model fixed --ne 0.03

5 kpc + 银心方向 NE2001：
    python dist_dm_convert.py --distance 5 --model ne2001 --l 0 --b 0

红移 z=0.5 的 IGM 贡献（约 500 pc/cm^3）：
    python dist_dm_convert.py --distance 1000 --dist-unit Mpc \\
        --z 0.5 --cosmic --model fixed --ne 0

反向：DM=30，无 l/b，固定 ne=0.03：
    python dist_dm_convert.py --dm 30 --model fixed --ne 0.03 --reverse

注意
----
如果 --model 选了 ne2001/ymw16 但没有 --l --b，或没有 pygedm，
脚本会打印警告并自动退化为 fixed 模型（用 --ne 计算）。
"""

import argparse
import math
import sys


# ==================== 输出编码健壮性（Windows GBK 兜底） ====================
def _fix_streams():
    """让 stdout/stderr 遇到无法编码的字符时替换为 '?'，而不是崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")
        except Exception:
            pass


_fix_streams()


# ==================== 物理常数（CGS） ====================
C_CMS = 2.99792458e10              # 光速 cm/s
PC_CM = 3.0856775814913673e18      # 1 pc in cm
KPC_PC = 1.0e3                     # 1 kpc in pc
MPC_PC = 1.0e6                     # 1 Mpc in pc
G_CGS = 6.67430e-8                 # 引力常数 cm^3 g^-1 s^-2
M_P_CGS = 1.67262192e-24           # 质子质量 g

# ==================== 宇宙学参数 ====================
H0_KMS_MPC = 70.0                  # Hubble 常数 km/s/Mpc
OMEGA_M = 0.30
OMEGA_L = 0.70
OMEGA_B = 0.049
F_IGM = 0.83
MPC_CM = 3.08567758e24             # 1 Mpc in cm


# ==================== 工具：Quantity → float ====================
def _qval(x, target_unit=None):
    """把 astropy Quantity 安全转成 float。

    - 如果给了 target_unit 且 x 有 .to_value，先转换单位
    - 否则取 .value
    - 如果 x 已经是 float / int / bool / None，直接处理
    - 任何异常都返回 nan，不会崩
    """
    if x is None:
        return float('nan')

    # Quantity 路径
    if hasattr(x, "to_value"):
        if target_unit is not None:
            try:
                return float(x.to_value(target_unit))
            except Exception:
                pass
        try:
            return float(x.value)
        except Exception:
            return float('nan')

    # 普通标量 / bool
    try:
        return float(x)
    except Exception:
        return float('nan')


# ==================== 单位转换 ====================
def to_pc(distance, unit):
    u = unit.lower()
    if u == 'pc':
        return distance
    if u == 'kpc':
        return distance * KPC_PC
    if u == 'mpc':
        return distance * MPC_PC
    raise ValueError(f"未知距离单位：{unit}（支持 pc / kpc / Mpc）")


# ==================== 宇宙学 ====================
def E_of_z(z, Om=OMEGA_M, OL=OMEGA_L):
    return math.sqrt(Om * (1 + z) ** 3 + OL)


def cosmic_dm(z, H0=H0_KMS_MPC, Om=OMEGA_M, OL=OMEGA_L,
              Ob=OMEGA_B, f_igm=F_IGM, n_step=1000):
    """Inoue 2004 的 IGM 色散量，返回 pc/cm^3。

    DM_IGM = (3 c H0 Ob f_igm) / (8 pi G m_p) * integral_0^z (1+z')/E(z') dz'
    """
    if z <= 0:
        return 0.0

    H0_cgs = H0 * 1e5 / MPC_CM
    prefactor_cgs = (3 * C_CMS * H0_cgs * Ob * f_igm) / (
        8 * math.pi * G_CGS * M_P_CGS
    )
    prefactor_pc = prefactor_cgs / PC_CM   # 约 957 pc/cm^3

    total = 0.0
    for i in range(n_step):
        z1 = z * i / n_step
        z2 = z * (i + 1) / n_step
        f1 = (1 + z1) / E_of_z(z1, Om, OL)
        f2 = (1 + z2) / E_of_z(z2, Om, OL)
        total += 0.5 * (f1 + f2) * (z2 - z1)

    return prefactor_pc * total


# ==================== 银河 DM ====================
def galactic_dm_fixed(distance_pc, ne):
    return ne * distance_pc


def galactic_dm_pygedm(distance_pc, l, b, method):
    """dist_to_dm 返回 (DM, limit)。这里统一返回 (float pc/cm3, float limit)。"""
    import pygedm
    import astropy.units as u

    result = pygedm.dist_to_dm(l, b, distance_pc, method=method)
    if isinstance(result, tuple):
        dm_raw = result[0]
        limit_raw = result[1] if len(result) > 1 else None
    else:
        dm_raw = result
        limit_raw = None

    # DM 单位通常是 pc/cm^3；显式转换更稳
    dm = _qval(dm_raw, target_unit=u.pc / u.cm ** 3)
    limit = _qval(limit_raw)
    return dm, limit


def dm_to_dist_pygedm(dm, l, b, method):
    """dm_to_dist 返回 (dist, limit)。这里统一返回 (float pc, float limit)。"""
    import pygedm
    import astropy.units as u

    result = pygedm.dm_to_dist(l, b, dm, method=method)
    if isinstance(result, tuple):
        dist_raw = result[0]
        limit_raw = result[1] if len(result) > 1 else None
    else:
        dist_raw = result
        limit_raw = None

    # 距离单位可能是 pc 或 kpc；统一转 pc
    dist_pc = _qval(dist_raw, target_unit=u.pc)
    limit = _qval(limit_raw)
    return dist_pc, limit


# ==================== 模型解析 ====================
def resolve_model(args):
    m = (args.model or "fixed").lower()
    if m not in ("fixed", "ne2001", "ymw16"):
        raise ValueError(
            f"未知模型 {args.model!r}（支持 fixed / ne2001 / ymw16）"
        )
    if m in ("ne2001", "ymw16"):
        if args.l is None or args.b is None:
            print(
                f"[警告] --model {m} 需要 --l --b，"
                f"未提供时退化为 fixed 模型",
                file=sys.stderr,
            )
            return "fixed"
        try:
            import pygedm  # noqa: F401
        except ImportError:
            print(
                f"[警告] 未安装 pygedm，--model {m} 退化为 fixed 模型",
                file=sys.stderr,
            )
            print("        pip install pygedm", file=sys.stderr)
            return "fixed"
    return m


# ==================== 计算 ====================
def distance_to_dm(distance_pc, model, args):
    if model == "fixed":
        if args.ne is None:
            raise ValueError("fixed 模型需要 --ne")
        return galactic_dm_fixed(distance_pc, args.ne), "fixed", \
            {"ne": args.ne}
    elif model in ("ne2001", "ymw16"):
        dm, limit = galactic_dm_pygedm(
            distance_pc, args.l, args.b, model)
        return dm, model, {"l": args.l, "b": args.b, "limit": limit}
    raise ValueError(f"未知模型 {model}")


def dm_to_distance(dm, model, args):
    if model == "fixed":
        if args.ne is None or args.ne <= 0:
            raise ValueError("fixed 模型需要正的 --ne（当前 <= 0）")
        return dm / args.ne, "fixed", {"ne": args.ne}
    elif model in ("ne2001", "ymw16"):
        d, limit = dm_to_dist_pygedm(dm, args.l, args.b, model)
        return d, model, {"l": args.l, "b": args.b, "limit": limit}
    raise ValueError(f"未知模型 {model}")


# ==================== 输出（纯 ASCII，中文除外） ====================
def print_forward(args, distance_pc, dm_gal, dm_ig, model, info):
    print(f"输入距离：{args.distance} {args.dist_unit}"
          f"  =  {distance_pc:.6g} pc")
    print(f"模型：{model}")
    if model == "fixed":
        print(f"  ne = {info['ne']} cm^-3")
    elif model in ("ne2001", "ymw16"):
        print(f"  l = {info['l']:.2f} deg, b = {info['b']:.2f} deg")
        lim = info.get('limit', float('nan'))
        if not (isinstance(lim, float) and math.isnan(lim)):
            print(f"  pygedm limit flag = {int(lim)}")

    print(f"\n银河 DM：{dm_gal:>10.4f} pc/cm^3")
    dm_total = dm_gal
    if args.cosmic:
        dm_total += dm_ig
        print(f"IGM  DM (z={args.z})：{dm_ig:>10.4f} pc/cm^3")
    elif args.z is not None and args.z > 0:
        print(f"[提示] 给了 --z={args.z} 但未加 --cosmic，不计算 IGM")
    print(f"总 DM：   {dm_total:>10.4f} pc/cm^3")

    # 群延迟参考：420 vs 1400 MHz
    nu1_ghz, nu2_ghz = 0.42, 1.4
    dt_ms = 4.15 * dm_total * (
        1.0 / nu1_ghz ** 2 - 1.0 / nu2_ghz ** 2)
    print(f"\n(420-1400 MHz 群延迟约 {dt_ms:.2f} ms)")


def print_reverse(args, distance_pc, dm_gal, dm_ig, model, info):
    print(f"输入 DM：{args.dm:.4f} pc/cm^3")
    if args.cosmic:
        print(f"IGM  DM (z={args.z})：{dm_ig:.4f} pc/cm^3")
        print(f"银河 DM：{dm_gal:.4f} pc/cm^3")
    print(f"模型：{model}")
    if model == "fixed":
        print(f"  ne = {info['ne']} cm^-3")
    elif model in ("ne2001", "ymw16"):
        print(f"  l = {info['l']:.2f} deg, b = {info['b']:.2f} deg")
        lim = info.get('limit', float('nan'))
        if not (isinstance(lim, float) and math.isnan(lim)):
            print(f"  pygedm limit flag = {int(lim)}")

    print(f"\n输出距离：")
    print(f"  {distance_pc:.6g} pc")
    print(f"  {distance_pc / KPC_PC:.6g} kpc")
    print(f"  {distance_pc / MPC_PC:.6g} Mpc")


# ==================== CLI ====================
_NEG_VALUE_OPTS = {"--ne", "--l", "--b", "--z", "--distance", "--dm",
                   "--h0", "--om", "--ol"}


def _normalize_argv(argv):
    """把 '--opt -1.5' 改写成 '--opt=-1.5'，规避 argparse 把负数当选项。"""
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
        description="距离 <-> 色散量 (DM) 转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python dist_dm_convert.py --distance 1.0 --model fixed --ne 0.03
  python dist_dm_convert.py --distance 5 --model ne2001 --l 0 --b 0
  python dist_dm_convert.py --distance 1000 --dist-unit Mpc \\
      --z 0.5 --cosmic --model fixed --ne 0
  python dist_dm_convert.py --dm 30 --model fixed --ne 0.03 --reverse
""",
    )

    p.add_argument("--distance", type=float, default=None,
                   help="距离（正向输入，默认单位 kpc）")
    p.add_argument("--dist-unit", default="kpc",
                   choices=["pc", "kpc", "Mpc"],
                   help="距离单位（默认 kpc）")
    p.add_argument("--dm", type=float, default=None,
                   help="色散量 DM (pc/cm^3)，反向输入")
    p.add_argument("--reverse", action="store_true",
                   help="反向：DM -> 距离")
    p.add_argument("--model", default="fixed",
                   choices=["fixed", "ne2001", "ymw16"],
                   help="电子密度模型（默认 fixed）")
    p.add_argument("--ne", type=float, default=0.03,
                   help="固定电子密度 cm^-3，仅 fixed 模型（默认 0.03）")
    p.add_argument("--l", type=float, default=None,
                   help="银经 deg，ne2001/ymw16 需要")
    p.add_argument("--b", type=float, default=None,
                   help="银纬 deg，ne2001/ymw16 需要")
    p.add_argument("--z", type=float, default=None,
                   help="红移，用于 IGM 贡献")
    p.add_argument("--cosmic", action="store_true",
                   help="加上 IGM 贡献（需要 --z）")
    p.add_argument("--h0", type=float, default=H0_KMS_MPC,
                   help=f"H0 km/s/Mpc（默认 {H0_KMS_MPC}）")
    p.add_argument("--om", type=float, default=OMEGA_M,
                   help=f"Omega_m（默认 {OMEGA_M}）")
    p.add_argument("--ol", type=float, default=OMEGA_L,
                   help=f"Omega_L（默认 {OMEGA_L}）")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(argv)
    args = build_parser().parse_args(argv)

    if args.reverse:
        if args.dm is None:
            print("[错误] --reverse 需要 --dm", file=sys.stderr)
            return 1
        if args.dm < 0:
            print("[错误] --dm 不能为负", file=sys.stderr)
            return 1
    else:
        if args.distance is None:
            print("[错误] 需要 --distance", file=sys.stderr)
            return 1
        if args.distance < 0:
            print("[错误] --distance 不能为负", file=sys.stderr)
            return 1

    if args.cosmic and args.z is None:
        print("[错误] --cosmic 需要 --z", file=sys.stderr)
        return 1
    if args.z is not None and args.z < 0:
        print("[错误] --z 不能为负", file=sys.stderr)
        return 1

    try:
        model_used = resolve_model(args)
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    dm_ig = 0.0
    if args.cosmic and args.z is not None:
        try:
            dm_ig = cosmic_dm(
                args.z, H0=args.h0, Om=args.om, OL=args.ol)
        except Exception as e:
            print(f"[错误] 计算 IGM DM 失败：{e}", file=sys.stderr)
            return 1

    try:
        if args.reverse:
            _do_reverse(args, model_used, dm_ig)
        else:
            _do_forward(args, model_used, dm_ig)
    except (ValueError, RuntimeError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    return 0


def _do_forward(args, model_used, dm_ig):
    distance_pc = to_pc(args.distance, args.dist_unit)
    dm_gal, model, info = distance_to_dm(distance_pc, model_used, args)
    print_forward(args, distance_pc, dm_gal, dm_ig, model, info)


def _do_reverse(args, model_used, dm_ig):
    dm_total = args.dm
    if args.cosmic and dm_ig > 0:
        dm_gal = dm_total - dm_ig
        if dm_gal <= 0:
            print(
                "[警告] 总 DM 不足以支付 IGM 贡献，银河系 DM 视为 0",
                file=sys.stderr,
            )
            dm_gal = 0.0
    else:
        dm_gal = dm_total
    distance_pc, model, info = dm_to_distance(dm_gal, model_used, args)
    print_reverse(args, distance_pc, dm_gal, dm_ig, model, info)


if __name__ == "__main__":
    sys.exit(main())

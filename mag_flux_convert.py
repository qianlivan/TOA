#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mag_flux_convert.py
===================
星等 ↔ 流量密度转换（支持多测光系统 / 多零点体系）

依赖：仅标准库。

支持
----
零点体系（--system）：vega / ab / st
测光系统（--survey）：
    johnson   U B V R I                     (Vega)
    cousins   Rc Ic                         (Vega)
    sdss      u g r i z                     (AB)
    2mass     J H Ks                        (Vega)
    ukidss    Y J H K                       (Vega)
    gaia      G BP RP                       (AB)
    wise      W1 W2 W3 W4                   (Vega)
    galex     FUV NUV                       (AB)
    panstarrs g r i z y                     (AB)
    tycho     BT VT                         (Vega)
    stromgren u v b y                       (Vega)
    spitzer   IRAC1 IRAC2 IRAC3 IRAC4       (Vega)
    kepler    Kepler                        (AB)

用法
----
正向（星等 → 流量）：
    python mag_flux_convert.py --mag 10 --band V
    python mag_flux_convert.py --mag 20 --survey sdss --band r --unit uJy
    python mag_flux_convert.py --mag 15 --zero-point 4000
    python mag_flux_convert.py --mag 10 --survey 2mass --band Ks
    python mag_flux_convert.py --mag 12 --band r --system vega   # 强制 Vega 体系（需波段有 Vega 零点）

反向（流量 → 星等）：
    python mag_flux_convert.py --flux 3631 --unit Jy --survey sdss --band r --reverse

查看内置波段：
    python mag_flux_convert.py --list-surveys
    python mag_flux_convert.py --list-bands
"""

import argparse
import math
import sys


# ==================== 常量 ====================
AB_ZERO_JY = 3631.0
ST_ZERO_F_LAMBDA = 3.631e-9
JY_IN_CGS = 1e-23
C_ANGSTROM_PER_S = 2.99792458e18
FNU_TO_FLAMBDA_FACTOR = JY_IN_CGS * C_ANGSTROM_PER_S


# ==================== 波段注册表 ====================
# key = "survey:band"，value = dict(f0_jy, system, wave, name)
# 说明：AB 系统 f0_jy=None（恒为 3631 Jy），Vega 系统填实际零点（Jy），wave 单位 Å。
BANDS = {
    # -------- Johnson (Vega) --------
    'johnson:U':  dict(f0_jy=1810.0, system='vega', wave=3650,  name='Johnson U'),
    'johnson:B':  dict(f0_jy=4260.0, system='vega', wave=4450,  name='Johnson B'),
    'johnson:V':  dict(f0_jy=3640.0, system='vega', wave=5510,  name='Johnson V'),
    'johnson:R':  dict(f0_jy=3080.0, system='vega', wave=6580,  name='Johnson R'),
    'johnson:I':  dict(f0_jy=2550.0, system='vega', wave=8060,  name='Johnson I'),
    # -------- Cousins (Vega) --------
    'cousins:Rc': dict(f0_jy=3080.0, system='vega', wave=6410,  name='Cousins Rc'),
    'cousins:Ic': dict(f0_jy=2550.0, system='vega', wave=7980,  name='Cousins Ic'),
    # -------- SDSS (AB) --------
    'sdss:u': dict(f0_jy=None, system='ab', wave=3560, name='SDSS u'),
    'sdss:g': dict(f0_jy=None, system='ab', wave=4680, name='SDSS g'),
    'sdss:r': dict(f0_jy=None, system='ab', wave=6165, name='SDSS r'),
    'sdss:i': dict(f0_jy=None, system='ab', wave=7480, name='SDSS i'),
    'sdss:z': dict(f0_jy=None, system='ab', wave=8930, name='SDSS z'),
    # -------- 2MASS (Vega) --------
    '2mass:J':  dict(f0_jy=1594.0, system='vega', wave=12350, name='2MASS J'),
    '2mass:H':  dict(f0_jy=1024.0, system='vega', wave=16620, name='2MASS H'),
    '2mass:Ks': dict(f0_jy=666.7,  system='vega', wave=21590, name='2MASS Ks'),
    # -------- UKIDSS (Vega, 近似) --------
    'ukidss:Y': dict(f0_jy=2023.0, system='vega', wave=10305, name='UKIDSS Y'),
    'ukidss:J': dict(f0_jy=1594.0, system='vega', wave=12483, name='UKIDSS J'),
    'ukidss:H': dict(f0_jy=1024.0, system='vega', wave=16313, name='UKIDSS H'),
    'ukidss:K': dict(f0_jy=666.7,  system='vega', wave=22010, name='UKIDSS K'),
    # -------- Gaia (AB) --------
    'gaia:G':  dict(f0_jy=None, system='ab', wave=6730, name='Gaia G'),
    'gaia:BP': dict(f0_jy=None, system='ab', wave=5320, name='Gaia BP'),
    'gaia:RP': dict(f0_jy=None, system='ab', wave=7970, name='Gaia RP'),
    # -------- WISE (Vega) --------
    'wise:W1': dict(f0_jy=309.54,  system='vega', wave=33526,  name='WISE W1'),
    'wise:W2': dict(f0_jy=171.787, system='vega', wave=46028,  name='WISE W2'),
    'wise:W3': dict(f0_jy=31.674,  system='vega', wave=115608, name='WISE W3'),
    'wise:W4': dict(f0_jy=8.363,   system='vega', wave=220883, name='WISE W4'),
    # -------- GALEX (AB) --------
    'galex:FUV': dict(f0_jy=None, system='ab', wave=1538, name='GALEX FUV'),
    'galex:NUV': dict(f0_jy=None, system='ab', wave=2316, name='GALEX NUV'),
    # -------- Pan-STARRS1 (AB) --------
    'panstarrs:g': dict(f0_jy=None, system='ab', wave=4810, name='PS1 g'),
    'panstarrs:r': dict(f0_jy=None, system='ab', wave=6170, name='PS1 r'),
    'panstarrs:i': dict(f0_jy=None, system='ab', wave=7520, name='PS1 i'),
    'panstarrs:z': dict(f0_jy=None, system='ab', wave=8660, name='PS1 z'),
    'panstarrs:y': dict(f0_jy=None, system='ab', wave=9620, name='PS1 y'),
    # -------- Tycho (Vega, 近似) --------
    'tycho:BT': dict(f0_jy=4175.0, system='vega', wave=4200, name='Tycho BT'),
    'tycho:VT': dict(f0_jy=4050.0, system='vega', wave=5270, name='Tycho VT'),
    # -------- Strömgren (Vega, 近似) --------
    'stromgren:u': dict(f0_jy=4186.0, system='vega', wave=3450, name='Strömgren u'),
    'stromgren:v': dict(f0_jy=3880.0, system='vega', wave=4110, name='Strömgren v'),
    'stromgren:b': dict(f0_jy=4080.0, system='vega', wave=4670, name='Strömgren b'),
    'stromgren:y': dict(f0_jy=3510.0, system='vega', wave=5470, name='Strömgren y'),
    # -------- Spitzer IRAC (Vega) --------
    'spitzer:IRAC1': dict(f0_jy=277.5, system='vega', wave=35500, name='Spitzer IRAC 3.6μm'),
    'spitzer:IRAC2': dict(f0_jy=179.7, system='vega', wave=45000, name='Spitzer IRAC 4.5μm'),
    'spitzer:IRAC3': dict(f0_jy=116.6, system='vega', wave=58000, name='Spitzer IRAC 5.8μm'),
    'spitzer:IRAC4': dict(f0_jy=63.1,  system='vega', wave=80000, name='Spitzer IRAC 8.0μm'),
    # -------- Kepler (AB) --------
    'kepler:Kepler': dict(f0_jy=None, system='ab', wave=6500, name='Kepler'),
}

SURVEYS = ['johnson', 'cousins', 'sdss', '2mass', 'ukidss', 'gaia', 'wise',
           'galex', 'panstarrs', 'tycho', 'stromgren', 'spitzer', 'kepler']


# ==================== 波段查询 ====================
def lookup_band(band, survey=None):
    """返回 (survey, band, info)。band 可写成 'johnson:V' 形式。"""
    if not band:
        raise ValueError("空波段名")

    if ':' in band:
        s, b = band.split(':', 1)
        if survey and survey.lower() != s.lower():
            raise ValueError(f"--band 中已含 survey={s}，与 --survey={survey} 冲突")
        survey, band = s, b

    if survey:
        s = survey.lower()
        if s not in SURVEYS:
            raise ValueError(f"未知测光系统 {survey!r}；可用 --list-surveys 查看")
        key = f"{s}:{band}"
        if key in BANDS:
            return s, band, BANDS[key]
        for k in BANDS:
            ks, kb = k.split(':', 1)
            if ks == s and kb.lower() == band.lower():
                return ks, kb, BANDS[k]
        known = sorted(k.split(':', 1)[1] for k in BANDS if k.startswith(s + ':'))
        raise ValueError(f"测光系统 {s} 中没有波段 {band!r}；可用：{known}")

    # 无 survey：全局精确匹配
    matches = [k for k in BANDS if k.split(':', 1)[1] == band]
    if len(matches) == 1:
        ks, kb = matches[0].split(':', 1)
        return ks, kb, BANDS[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"波段 {band!r} 出现在多个测光系统：{matches}，请用 --survey 指定")

    # 大小写不敏感 fallback
    matches = [k for k in BANDS if k.split(':', 1)[1].lower() == band.lower()]
    if len(matches) == 1:
        ks, kb = matches[0].split(':', 1)
        return ks, kb, BANDS[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"波段 {band!r} 有多个大小写不敏感匹配：{matches}，请用 --survey 指定")
    raise ValueError(f"未知波段 {band!r}；可用 --list-bands 查看")


def list_surveys():
    print("内置测光系统：")
    for s in SURVEYS:
        n = sum(1 for k in BANDS if k.startswith(s + ':'))
        print(f"  {s:<10s} {n} 个波段")


def list_bands():
    by_survey = {}
    for k, v in BANDS.items():
        s, b = k.split(':', 1)
        by_survey.setdefault(s, []).append((b, v))

    for s in SURVEYS:
        if s not in by_survey:
            continue
        print(f"\n[{s}]")
        for b, v in sorted(by_survey[s]):
            if v['system'] == 'ab':
                f0_str = f"{AB_ZERO_JY:>8.2f} Jy (AB)"
            else:
                f0_str = f"{v['f0_jy']:>8.2f} Jy (Vega)"
            print(f"  {b:>8s}   f0={f0_str}   λ_eff={v['wave']:>7d} Å   ({v['name']})")


# ==================== 基本公式 ====================
def mag_to_fnu_jy(m, f0_jy):
    return f0_jy * 10.0 ** (-0.4 * m)


def fnu_jy_to_mag(f_jy, f0_jy):
    if f_jy <= 0:
        raise ValueError(f"流量必须为正数，当前为 {f_jy}")
    return -2.5 * math.log10(f_jy / f0_jy)


def jy_to_flambda(f_jy, wave_ang):
    if wave_ang <= 0:
        raise ValueError("波长必须为正")
    return f_jy * FNU_TO_FLAMBDA_FACTOR / wave_ang ** 2


def flambda_to_st_mag(f_lambda):
    if f_lambda <= 0:
        raise ValueError("Fλ 必须为正")
    return -2.5 * math.log10(f_lambda / ST_ZERO_F_LAMBDA)


def fnu_to_ab_mag(f_jy):
    return fnu_jy_to_mag(f_jy, AB_ZERO_JY)


def fnu_to_st_mag(f_jy, wave_ang):
    return flambda_to_st_mag(jy_to_flambda(f_jy, wave_ang))


# ==================== 单位 ====================
def unit_to_jy(value, unit):
    u = unit.lower()
    if u == 'jy':
        return value
    if u == 'mjy':
        return value * 1e-3
    if u in ('ujy', 'μjy', 'microjy'):
        return value * 1e-6
    raise ValueError(f"不支持的单位：{unit}（支持 Jy / mJy / uJy）")


def jy_to_unit(f_jy, unit):
    u = unit.lower()
    if u == 'jy':
        return f_jy
    if u == 'mjy':
        return f_jy * 1e3
    if u in ('ujy', 'μjy', 'microjy'):
        return f_jy * 1e6
    raise ValueError(f"不支持的单位：{unit}（支持 Jy / mJy / uJy）")


# ==================== 解析零点 / 波长 ====================
def resolve_wavelength(args, band_info):
    """优先 --wavelength，其次 band_info['wave']，都没有则 None。"""
    if args.wavelength is not None:
        w = float(args.wavelength)
        if w <= 0:
            raise ValueError("--wavelength 必须为正")
        return w
    if band_info and band_info.get('wave'):
        return float(band_info['wave'])
    return None


def resolve_zero_point(args):
    """返回 dict(f0_jy, system, survey, band, wave, desc)。

    优先级：--zero-point > --system > (--survey, --band) 自身属性 > 默认 AB
    """
    # 1) 自定义零点
    if args.zero_point is not None:
        f0 = float(args.zero_point)
        if f0 <= 0:
            raise ValueError("--zero-point 必须为正")
        survey = args.survey
        band = args.band
        wave = resolve_wavelength(args, None)
        return dict(f0_jy=f0, system='custom', survey=survey, band=band,
                    wave=wave, desc=f"自定义零点（{f0} Jy）")

    # 2) 查询波段
    survey, band, info = None, None, None
    if args.band:
        survey, band, info = lookup_band(args.band, args.survey)
    elif args.survey:
        raise ValueError("--survey 需要配合 --band 使用")

    # 3) --system 覆盖
    if args.system:
        s = args.system.lower()
        if s == 'ab':
            wave = resolve_wavelength(args, info)
            return dict(f0_jy=AB_ZERO_JY, system='ab', survey=survey,
                        band=band, wave=wave,
                        desc=f"AB 系统（{AB_ZERO_JY} Jy）")
        if s == 'vega':
            if info and info.get('f0_jy'):
                return dict(f0_jy=info['f0_jy'], system='vega',
                            survey=survey, band=band, wave=info['wave'],
                            desc=f"Vega {info['name']} 零点（{info['f0_jy']} Jy）")
            raise ValueError(f"波段 {band!r} 没有 Vega 零点，请改 --system ab 或 --zero-point")
        if s == 'st':
            wave = resolve_wavelength(args, info)
            if wave is None:
                raise ValueError("ST 系统需要 --wavelength 或带有效波长的波段")
            f0 = ST_ZERO_F_LAMBDA * wave ** 2 / FNU_TO_FLAMBDA_FACTOR
            return dict(f0_jy=f0, system='st', survey=survey, band=band,
                        wave=wave, desc=f"ST 系统（λ={wave:.0f} Å）")
        raise ValueError(f"未知系统 {args.system!r}（vega / ab / st）")

    # 4) 波段自身属性
    if info:
        if info['system'] == 'ab':
            return dict(f0_jy=AB_ZERO_JY, system='ab', survey=survey,
                        band=band, wave=info['wave'],
                        desc=f"AB 系统（{AB_ZERO_JY} Jy）")
        return dict(f0_jy=info['f0_jy'], system='vega', survey=survey,
                    band=band, wave=info['wave'],
                    desc=f"Vega {info['name']} 零点（{info['f0_jy']} Jy）")

    # 5) 默认 AB
    return dict(f0_jy=AB_ZERO_JY, system='ab', survey=None, band=None,
                wave=None, desc=f"AB 系统（{AB_ZERO_JY} Jy，默认）")


# ==================== 输出 ====================
def _report_header(ctx, prefix=""):
    print(f"{prefix}零点体系：{ctx['system']}")
    if ctx['survey']:
        print(f"  测光系统：{ctx['survey']}")
    if ctx['band']:
        print(f"  波段：{ctx['band']}")
    if ctx['wave']:
        print(f"  有效波长：{ctx['wave']:.1f} Å")
    print(f"  说明：{ctx['desc']}")


def _report_equivalent_mags(f_jy, wave, ctx):
    entries = []
    if ctx['system'] != 'ab':
        entries.append(("AB", fnu_to_ab_mag(f_jy)))
    if wave and ctx['system'] != 'st':
        entries.append(("ST", fnu_to_st_mag(f_jy, wave)))
    if ctx['survey'] and ctx['band']:
        key = f"{ctx['survey']}:{ctx['band']}"
        if key in BANDS and BANDS[key].get('f0_jy') and ctx['system'] != 'vega':
            entries.append((f"Vega({ctx['band']})",
                            fnu_jy_to_mag(f_jy, BANDS[key]['f0_jy'])))
    if not entries:
        return
    print("\n等效星等：")
    for name, val in entries:
        print(f"  {name:>12s} = {val:.6f}")


def _report_flambda(f_jy, wave):
    if wave is None:
        print("  （无波长信息，无法给出 Fλ / ST 星等）")
        return
    f_lambda = jy_to_flambda(f_jy, wave)
    print(f"\nFλ（λ = {wave:.1f} Å）：")
    print(f"  {f_lambda:.6e} erg/s/cm²/Å")


def print_mag_to_flux(args):
    m = float(args.mag)
    ctx = resolve_zero_point(args)
    wave = ctx['wave']
    f_jy = mag_to_fnu_jy(m, ctx['f0_jy'])

    print(f"输入星等：m = {m:.6f}")
    _report_header(ctx)
    print()
    f_unit = jy_to_unit(f_jy, args.unit)
    print("输出流量密度（Fν）：")
    print(f"  {f_unit:.6e} {args.unit}")
    if args.unit.lower() != 'jy':
        print(f"  = {f_jy:.6e} Jy")
    _report_flambda(f_jy, wave)
    _report_equivalent_mags(f_jy, wave, ctx)


def print_flux_to_mag(args):
    if args.flux is None or args.flux == '':
        raise ValueError("反向转换需要 --flux")
    f_in = float(args.flux)
    f_jy = unit_to_jy(f_in, args.unit)
    ctx = resolve_zero_point(args)
    wave = ctx['wave']
    m = fnu_jy_to_mag(f_jy, ctx['f0_jy'])

    print(f"输入流量：{f_in:.6e} {args.unit}")
    if args.unit.lower() != 'jy':
        print(f"        = {f_jy:.6e} Jy")
    _report_header(ctx)
    print(f"\n输出星等：m = {m:.6f}  [{ctx['system']}]")
    _report_flambda(f_jy, wave)
    _report_equivalent_mags(f_jy, wave, ctx)


# ==================== CLI ====================
_NEG_VALUE_OPTS = {'--mag', '--flux'}


def _normalize_argv(argv):
    out = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok in _NEG_VALUE_OPTS and i + 1 < n:
            nxt = argv[i + 1]
            if nxt.startswith('-') and not nxt.startswith('--'):
                out.append(f"{tok}={nxt}")
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="星等 ↔ 流量密度转换（多测光系统 / 多零点体系）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python mag_flux_convert.py --mag 10 --band V
  python mag_flux_convert.py --mag 20 --survey sdss --band r --unit uJy
  python mag_flux_convert.py --mag 10 --survey 2mass --band Ks
  python mag_flux_convert.py --mag 12 --band r --system vega
  python mag_flux_convert.py --flux 3631 --unit Jy --survey sdss --band r --reverse
  python mag_flux_convert.py --mag 15 --zero-point 4000
  python mag_flux_convert.py --mag 10 --system st --wavelength 5510
  python mag_flux_convert.py --list-surveys
  python mag_flux_convert.py --list-bands
""",
    )
    p.add_argument("--mag", default=None, help="星等（正向输入）")
    p.add_argument("--flux", default=None, help="流量密度（反向输入）")
    p.add_argument("--unit", default="Jy", help="流量单位：Jy / mJy / uJy（默认 Jy）")
    p.add_argument("--system", default=None,
                   help="零点体系：vega / ab / st（覆盖波段默认）")
    p.add_argument("--survey", default=None,
                   help="测光系统：" + " / ".join(SURVEYS))
    p.add_argument("--band", default=None,
                   help="波段名，可写 'johnson:V' 形式；不给 survey 时全局匹配")
    p.add_argument("--zero-point", default=None,
                   help="自定义零点流量（Jy），优先级最高")
    p.add_argument("--wavelength", default=None,
                   help="有效波长（Å），用于 Fλ / ST 换算（覆盖波段默认值）")
    p.add_argument("--reverse", action="store_true", help="反向：流量 → 星等")
    p.add_argument("--list-surveys", action="store_true", help="列出内置测光系统")
    p.add_argument("--list-bands", action="store_true", help="列出全部波段及其零点、有效波长")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_surveys:
        list_surveys()
        return 0
    if args.list_bands:
        list_bands()
        return 0

    try:
        if args.reverse:
            if args.flux is None or args.flux == '':
                parser.error("反向转换需要 --flux")
            print_flux_to_mag(args)
        else:
            if args.mag is None or args.mag == '':
                parser.error("正向转换需要 --mag")
            print_mag_to_flux(args)
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

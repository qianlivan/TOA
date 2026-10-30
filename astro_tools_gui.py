#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天文工具 GUI - 坐标 / 星等 / 距离-色散 / FITS 天图 / 模拟数据 / QuickCube / SpecFit / 参数检查

内置脚本（与本文件同目录）：
    radec2glongglat.py     赤道坐标 ↔ 银道坐标
    mag_flux_convert.py    星等 ↔ 流量密度（14 个测光系统）
    dist_dm_convert.py     距离 ↔ 色散量
    skymap_fits.py         FITS 天图绘制（WCS + 可选球面投影）
    make_mock_sky_fits.py  生成模拟全天 2D FITS 天图
    make_mock_cube_fits.py 生成模拟 3D 谱线 datacube
    quickcube.py           2D 图像 / 3D cube 交互查看 + 坏点标记
    specfit.py             3D cube 谱线高斯拟合（自动 / 交互 / BIC 自适应）
    inspect_specfit.py     参数 FITS 检查（列表 / 面板 / truth 对比）

扩展方式：
    - register_script(name, filename, description)
    - add_script_tab(...)
    - add_quick_button(...)
    - add_quick_sequence(...)
    - create_extensions()
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import sys
import os
import threading
import queue


# ================== 配置 ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VERSION = "1.0"

SCRIPTS = {
    'radec2glongglat': 'radec2glongglat.py',
    'mag_flux':        'mag_flux_convert.py',
    'dist_dm':         'dist_dm_convert.py',
    'skymap':          'skymap_fits.py',
    'make_mock':       'make_mock_sky_fits.py',
    'make_mock_cube':  'make_mock_cube_fits.py',
    'quickcube':       'quickcube.py',
    'specfit':         'specfit.py',
    'inspect_specfit': 'inspect_specfit.py',
}


# ================== 帮助文本 ==================
HELP_OPERATIONS = """\
天文工具集  -  常用操作
============================================================

▌选项卡 1：赤道 ↔ 银道
------------------------------------------------------------
  用途：赤道坐标 (RA, Dec) 与银道坐标 (l, b) 互转（J2000）。
  步骤：
    1) 在「RA」和「Dec」填写赤道坐标（支持 hms/dms 或十进制度）
    2) 直接点「转换」，日志区显示对应的 l、b
    3) 反向：填 l 和 b，勾选「反向：银道 → 赤道」，再点「转换」
  小技巧：
    - 输入十进制度时勾选「输入为十进制度」可跳过 hms 解析
    - 勾选「输出为十进制度」只看数字，方便复制到别的软件
  示例输入：RA=08h35m20.61125s  Dec=-45d10m34.875s
  预期输出：l≈263.55°, b≈-2.79°  （Vela 脉冲星）

▌选项卡 2：星等 ↔ 流量
------------------------------------------------------------
  用途：星等与流量密度互转，支持 14 个测光系统 / 60+ 波段。
  步骤：
    1) 「星等」填 m，「流量密度」留空
    2) 「零点体系」选 auto/vega/ab/st
    3) 「测光系统」选 johnson/sdss/2mass/...
    4) 「波段」下拉选具体滤光片（写成 survey:band 形式）
    5) 点「转换」
  反向：勾选「反向：流量 → 星等」，填「流量密度」，「星等」留空
  提示：
    - 不确定波段怎么写？点「列出全部波段」查看内置表
    - 自定义零点：直接在「自定义零点 (Jy)」填数值即可
    - AB / ST 系统需要有效波长；已知波段自动带出，未知波段手动填

▌选项卡 3：距离 ↔ 色散量
------------------------------------------------------------
  用途：银河系距离和 DM 的换算，支持 3 种电子密度模型。
  步骤：
    1) 填「距离」和「距离单位」(pc/kpc/Mpc)
    2) 「电子密度模型」下拉选择：
       - fixed  ：固定 ne，只需「固定电子密度」(默认 0.03 cm^-3)
       - ne2001 ：需要同时填「银经 l」「银纬 b」，并安装 pygedm
       - ymw16  ：同上
    3) 点「转换」输出 DM
  反向：勾选「反向：DM → 距离」
  退化警告：
    - 若选 ne2001/ymw16 但缺 l/b 或没装 pygedm，
      GUI 会弹窗问你「是否继续？」
      选「是」→ 自动退化为 fixed（用 --ne 计算）
      选「否」→ 回到 GUI 补填参数
    - 退化后日志里会打印 [警告] 行
  宇宙学贡献：
    - 勾「包含宇宙学贡献」需同时填「红移」
    - IGM 部分按 Inoue 2004 公式计算
    - 反向时若总 DM 不足以支付 IGM，会警告并把银河 DM 置 0
  注意：
    - 银河内 DM 用 NE2001 效果较好；z > 0.1 后才需要加 IGM
    - 真实脉冲星 DM 还取决于视线方向，模型只是近似

▌选项卡 4：FITS 天图
------------------------------------------------------------
  用途：把 FITS 2D 图像（HI 柱密度、射电连续谱等）画成天图。
  基本流程：
    1) 「FITS 文件」点「浏览...」选择 .fits
    2) 「投影」保持 native（用文件自带 WCS）
    3) 勾「对数归一化」，「颜色范围分位」填 0.5,99.5
    4) 点「绘制天图」，打开 skymap_fits.png
  想要更漂亮的全天图：
    - 「投影」选 aitoff / mollweide / hammer / lambert
    - 「坐标系」选 galactic（银道）或 equatorial（赤道）
    - 「中心经度」留空自动（银道=0，赤道=180）
  多波段对比：
    - FITS 文件填 a.fits,b.fits,c.fits
    - 面板标题填 A,B,C
    - 子图列数填 2
  调试：
    - 「列出 HDU 结构」查看文件里有哪些 HDU
    - 无数据？先点「生成模拟 2D FITS」生成 mock_hi.fits 练手

▌选项卡 5：QuickCube
------------------------------------------------------------
  用途：查看 2D / 3D FITS，鼠标标记坏区，保存 mask。
  步骤：
    1) 「FITS 文件」填 cube.fits（可逗号分隔多个）
    2) 「Mask 输出」填要保存的路径，例如 cube_mask.fits
    3) 点「启动交互窗口」
  交互窗口键盘：
    r   进入/退出标记模式（进入后拖拽鼠标画矩形）
    u   撤销最后一个矩形
    s   立即保存 mask
    c   清空所有标记
    n   下一个切片（3D cube）
    p   上一个切片
    g   切换网格
    q   退出（会问是否保存）
  多文件 RGB 合成：
    - 填 3 个文件，叠加模式选 rgb
    - 例如 J/H/K 三波段合成一张伪彩色图
  注意：
    - mask 保存为 uint8 FITS，1 = 坏像素
    - RGB 模式下 mask 无意义（没有单一原始数据对应）

▌选项卡 6：SpecFit
------------------------------------------------------------
  用途：对 cube 每个像素做谱线高斯拟合，输出参数 FITS + PNG。
  快速上手：
    1) 无 cube？点「生成模拟 cube」自动生成 mock_cube.fits
       （同时生成 mock_cube_mom0.fits 和 mock_cube_truth.fits）
    2) 「成分数」选 auto，「auto 最大成分数」填 3~6
    3) 点「自动批量拟合」，等日志完成
    4) 点「查看参数 FITS」看每参数统计 + 面板大图
  拟合模式：
    - 固定成分数：成分数下拉选 1/2/3/...，所有像素统一 ncomp
    - BIC 自适应：成分数选 auto，每像素独立选 1~max-ncomp 的最优
  交互模式：
    - 点「交互查看」弹窗，点积分图任一像素
    - 右侧显示谱线 + 总拟合 + 每条分量（虚线，标注 μ/FWHM）
  查看 cube：
    - 点「查看 cube 信息」打印 shape、速度轴、BUNIT
    - 速度范围想裁剪：填「速度范围」例如 -200,200
  参数文件：
    - 输出 specfit_params.fits，多 HDU
    - HDU 名 = amp1 / v1 / sigma1 / ... / c0 / chi2 / ncomp
    - 「查看参数 FITS」一键检查
    - 如果 mock_cube_truth.fits 存在，自动对比并出残差图

▌快捷栏（底栏）
------------------------------------------------------------
  清空日志
  示例：RA/Dec → l/b                    （坐标转换演示）
  示例：Vega V=10 → 流量                （星等转换演示）
  模拟 FITS → Aitoff 天图               （2D 模拟 → 天图）
  模拟 FITS → QuickCube                 （2D 模拟 → 查看器）
  模拟 cube → SpecFit → 参数检查        （3D 模拟 → 拟合 → 检查）

  任何时候想清空日志点「清空日志」即可。

▌菜单栏
------------------------------------------------------------
  文件 → 清空运行日志 / 退出
  工具 → 快速跳到任一选项卡
  帮助 → 常用操作 / 常用命令 / 关于

  帮助窗口都带「复制全部」按钮，方便保存到文件。

▌常见问题
------------------------------------------------------------
  Q: 点按钮没反应 / 卡住？
  A: 长时间任务在后台线程执行，GUI 仍可用；状态栏会显示「运行 ...」

  Q: 日志里出现「找不到脚本」？
  A: 把对应 .py 和 astro_tools_gui.py 放在同一目录

  Q: 画出来是直角长条，想变圆？
  A: FITS 天图选项卡里，投影选 aitoff/mollweide，坐标系选 galactic

  Q: SpecFit 报「unrecognized arguments: --auto-ncomp」？
  A: 你的 specfit.py 是旧版，更新到最新版即可

  Q: 距离↔色散量选了 ne2001 但结果和 fixed 一样？
  A: 缺 l/b 或没装 pygedm 时脚本会自动退化，检查日志里的 [警告] 行，
     或补填 l/b、pip install pygedm

  Q: 生成速度很慢？
  A: 默认尺寸已经够测试用；真要跑大 cube：
     make_mock_cube_fits.py --shape 128 360 720
     specfit.py --nproc 4
"""

HELP_COMMANDS = """\
天文工具集  -  常用命令
============================================================
所有脚本都支持 --help 查看参数说明。
路径均为相对当前目录；在 GUI 同目录运行可直接用。

▌1. radec2glongglat.py  （坐标转换）
------------------------------------------------------------
正向（赤道 → 银道）：
  python radec2glongglat.py --ra 08h35m20.61125s --dec=-45d10m34.875s

十进制度输入：
  python radec2glongglat.py --ra 128.8359 --dec -45.1764 --decimal-in

反向（银道 → 赤道）：
  python radec2glongglat.py --glon 262.9 --glat -4.0 --reverse

仅输出十进制度：
  python radec2glongglat.py --ra 08h35m20.6s --dec=-45d10m34.9s --decimal-out

▌2. mag_flux_convert.py  （星等 ↔ 流量）
------------------------------------------------------------
Vega V=10 → 流量：
  python mag_flux_convert.py --mag 10 --band johnson:V

SDSS r=20 → µJy：
  python mag_flux_convert.py --mag 20 --survey sdss --band r --unit uJy

反向：流量 → 星等：
  python mag_flux_convert.py --flux 3631 --unit Jy --band sdss:r --reverse

自定义零点：
  python mag_flux_convert.py --mag 15 --zero-point 4000

列出全部内置波段：
  python mag_flux_convert.py --list-bands

▌3. dist_dm_convert.py  （距离 ↔ 色散量）
------------------------------------------------------------
1 kpc 固定 ne：
  python dist_dm_convert.py --distance 1.0 --dist-unit kpc --model fixed --ne 0.03

银心方向 5 kpc NE2001（需 pip install pygedm）：
  python dist_dm_convert.py --distance 5 --model ne2001 --l 0 --b 0

反向 DM=30 固定 ne：
  python dist_dm_convert.py --dm 30 --model fixed --ne 0.03 --reverse

z=0.5 加 IGM 贡献：
  python dist_dm_convert.py --distance 1000 --dist-unit Mpc \\
      --z 0.5 --cosmic --model fixed --ne 0

▌4. skymap_fits.py  （FITS 天图）
------------------------------------------------------------
自带 WCS 直角图：
  python skymap_fits.py --input HI4PI.fits --log --cmap viridis --no-show

全天 Aitoff 银道图（推荐）：
  python skymap_fits.py --input HI4PI.fits --projection aitoff \\
      --coordsys galactic --log --cmap viridis --no-show

Mollweide 赤道图（中心 RA=180）：
  python skymap_fits.py --input HI4PI.fits --projection mollweide \\
      --coordsys equatorial --center-lon 180 --log --no-show

矩形重居中到 l=0：
  python skymap_fits.py --input HI4PI.fits --projection rect \\
      --coordsys galactic --center-lon 0 --no-show

多文件多面板：
  python skymap_fits.py --inputs U.fits,B.fits,V.fits,R.fits \\
      --labels U,B,V,R --projection aitoff --ncols 2 --no-show

查看 HDU 结构：
  python skymap_fits.py --input x.fits --list-hdu

▌5. make_mock_sky_fits.py  （模拟 2D 天图）
------------------------------------------------------------
默认：
  python make_mock_sky_fits.py --output mock_hi.fits

高分辨率 + 更多点源：
  python make_mock_sky_fits.py --shape 720 1440 --n-sources 1000 \\
      --output mock_hi_big.fits

▌6. make_mock_cube_fits.py  （模拟 3D 谱线 cube）
------------------------------------------------------------
默认（64ch × 180 × 360，2 个高斯成分）：
  python make_mock_cube_fits.py --output mock_cube.fits

同时输出 Moment-0 和真值：
  python make_mock_cube_fits.py --output mock_cube.fits \\
      --moment0 mock_cube_mom0.fits \\
      --save-truth mock_cube_truth.fits

较大 cube，含旋转速度场：
  python make_mock_cube_fits.py --shape 128 360 720 --v-disk 150 \\
      --output mock_cube_big.fits

单高斯（测 ncomp=1）：
  python make_mock_cube_fits.py --ncomp-max 1 --output cube1.fits

▌7. quickcube.py  （查看 + 标记坏点）
------------------------------------------------------------
直接打开（3D cube）：
  python quickcube.py --input cube.fits --slice 5 --mask-output mask.fits

三个文件 RGB 合成：
  python quickcube.py --inputs j.fits,h.fits,k.fits --stack rgb

对数显示：
  python quickcube.py --input hi.fits --log --percentile 1,99 \\
      --mask-output hi_mask.fits

仅测试加载（不弹窗）：
  python quickcube.py --input cube.fits --no-show

▌8. specfit.py  （谱线高斯拟合）
------------------------------------------------------------
单高斯批量拟合：
  python specfit.py --input cube.fits --ncomp 1 \\
      --param-output params.fits --no-show

双高斯 + 速度裁剪 + 4 进程：
  python specfit.py --input cube.fits --ncomp 2 \\
      --velocity-range -200,200 --snr-min 3 --nproc 4 \\
      --param-output params.fits --no-show

BIC 自适应成分数（推荐）：
  python specfit.py --input cube.fits \\
      --auto-ncomp --max-ncomp 6 --snr-min 4 --nproc 4 \\
      --param-output params.fits --plot-output maps --no-show

交互查看：
  python specfit.py --input cube.fits --mode interactive

查看 cube 信息：
  python specfit.py --input cube.fits --info

▌9. inspect_specfit.py  （参数 FITS 检查）
------------------------------------------------------------
只看结构 + 统计：
  python inspect_specfit.py --input params.fits

同时输出参数面板大图：
  python inspect_specfit.py --input params.fits --plot panel.png

与 truth 对比：
  python inspect_specfit.py --input params.fits \\
      --compare mock_cube_truth.fits --plot panel.png

▌10. 组合流水线
------------------------------------------------------------
一键模拟 + 拟合 + 检查：
  python make_mock_cube_fits.py --output mock_cube.fits \\
      --moment0 mock_cube_mom0.fits --save-truth mock_cube_truth.fits
  python specfit.py --input mock_cube.fits \\
      --auto-ncomp --max-ncomp 3 --param-output mock_cube_params.fits \\
      --plot-output mock_cube_maps --no-show
  python inspect_specfit.py --input mock_cube_params.fits \\
      --plot mock_cube_maps_panel.png --compare mock_cube_truth.fits

  在 GUI 底栏点「模拟 cube → SpecFit → 参数检查」等价于上面三步。
"""

HELP_ABOUT = f"""\
天文工具集  Astro Tools
============================================================

版本：{VERSION}
文件：astro_tools_gui.py
依赖：tkinter（标准库）, subprocess, threading

外部依赖：
  - numpy, scipy, matplotlib, astropy
    安装：pip install numpy scipy matplotlib astropy
  - pygedm（可选，用于 ne2001 / ymw16 模型）
    安装：pip install pygedm

内置脚本（同一目录）：
  radec2glongglat.py     赤道 ↔ 银道
  mag_flux_convert.py    星等 ↔ 流量
  dist_dm_convert.py     距离 ↔ 色散量
  skymap_fits.py         FITS 天图
  make_mock_sky_fits.py  模拟 2D 天图
  make_mock_cube_fits.py 模拟 3D cube
  quickcube.py           2D/3D 查看 + 坏点标记
  specfit.py             谱线高斯拟合
  inspect_specfit.py     参数 FITS 检查

扩展方式：
  见 create_extensions() 里的示例代码
  - register_script(...) 注册新脚本
  - add_script_tab(...) 声明式新增选项卡
  - add_quick_button(...) 快捷栏按钮
  - add_quick_sequence(...) 快捷栏流水线

帮助文档：
  菜单「帮助 → 常用操作」  中文操作手册
  菜单「帮助 → 常用命令」  各脚本 CLI 示例
"""


# ================== 参数描述辅助函数 ==================
def P(label, key, default="", arg=None, browse=None):
    return dict(label=label, key=key, default=default, arg=arg, browse=browse)


def F(label, key, default=False, arg=None):
    return dict(label=label, key=key, default=default, arg=arg)


def _arg_name(spec):
    return spec["arg"] or ("--" + spec["key"].replace("_", "-"))


# ================== 声明式选项卡 ==================
class ScriptTab:
    def __init__(self, app, title, script,
                 input_file=None, positional_input=True,
                 params=(), flags=(), button_text="运行",
                 notify=False, extra_ui=None, on_complete=None,
                 log_height=12):
        self.app = app
        self.script = script
        self.input_file = input_file
        self.positional_input = positional_input
        self.params = list(params)
        self.flags = list(flags)
        self.notify = notify
        self.on_complete = on_complete
        self.log_height = log_height
        self.vars = {}
        self.flag_vars = {}
        self.extra_args_provider = None

        self.frame = ttk.Frame(app.notebook)
        app.notebook.add(self.frame, text=title)
        self._build(button_text, extra_ui)

    def _build(self, button_text, extra_ui):
        frame = self.frame
        row = 0

        def add_field(spec, r):
            ttk.Label(frame, text=spec["label"]).grid(
                row=r, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=str(spec["default"]))
            self.vars[spec["key"]] = var
            ttk.Entry(frame, textvariable=var, width=40).grid(
                row=r, column=1, sticky=tk.W, padx=5, pady=2)
            if spec.get("browse"):
                ttk.Button(
                    frame, text="浏览...",
                    command=lambda v=var, m=spec["browse"]:
                        self.app.browse_file(v, mode=m)
                ).grid(row=r, column=2, padx=5)

        if self.input_file:
            add_field(self.input_file, row)
            row += 1

        for spec in self.params:
            add_field(spec, row)
            row += 1

        if self.flags:
            flag_frame = ttk.Frame(frame)
            flag_frame.grid(row=row, column=0, columnspan=3,
                            sticky=tk.W, padx=5, pady=2)
            for spec in self.flags:
                var = tk.BooleanVar(value=spec["default"])
                self.flag_vars[spec["key"]] = var
                ttk.Checkbutton(flag_frame, text=spec["label"],
                                variable=var).pack(side=tk.LEFT, padx=5)
            row += 1

        if extra_ui:
            extra_frame = ttk.Frame(frame)
            extra_frame.grid(row=row, column=0, columnspan=3,
                             sticky=tk.W + tk.E, padx=5, pady=2)
            extra_ui(self, extra_frame)
            row += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text=button_text,
                   command=self.run).pack(side=tk.LEFT, padx=5)
        row += 1

        self.log = scrolledtext.ScrolledText(frame, height=self.log_height,
                                             width=80)
        self.log.grid(row=row, column=0, columnspan=3,
                      padx=5, pady=5, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

    def collect_args(self):
        args = []

        if self.input_file:
            spec = self.input_file
            val = self.vars[spec["key"]].get().strip()
            if not val:
                raise ValueError(f"请填写：{spec['label']}")
            if self.positional_input:
                args.append(val)
            else:
                args.extend([_arg_name(spec), val])

        for spec in self.params:
            val = self.vars[spec["key"]].get().strip()
            if val != "":
                args.extend([_arg_name(spec), val])

        for spec in self.flags:
            if self.flag_vars[spec["key"]].get():
                args.append(_arg_name(spec))

        if self.extra_args_provider:
            args.extend(self.extra_args_provider())

        return args

    def run(self):
        try:
            args = self.collect_args()
        except ValueError as e:
            messagebox.showerror("错误", str(e))
            return
        self.app.run_script_async(self.script, args, self.log,
                                  on_complete=self.on_complete,
                                  notify=self.notify)


# ================== 主 GUI 类 ==================
class AstroToolsGUI:
    # ---- mag_flux 下拉 ----
    _SYSTEM_CHOICES = ["auto", "vega", "ab", "st"]

    _SURVEY_CHOICES = [
        "auto",
        "johnson", "cousins", "sdss", "2mass", "ukidss",
        "gaia", "wise", "galex", "panstarrs",
        "tycho", "stromgren", "spitzer", "kepler",
    ]

    _BAND_CHOICES = [
        "",
        "johnson:U", "johnson:B", "johnson:V", "johnson:R", "johnson:I",
        "cousins:Rc", "cousins:Ic",
        "sdss:u", "sdss:g", "sdss:r", "sdss:i", "sdss:z",
        "2mass:J", "2mass:H", "2mass:Ks",
        "ukidss:Y", "ukidss:J", "ukidss:H", "ukidss:K",
        "gaia:G", "gaia:BP", "gaia:RP",
        "wise:W1", "wise:W2", "wise:W3", "wise:W4",
        "galex:FUV", "galex:NUV",
        "panstarrs:g", "panstarrs:r", "panstarrs:i",
        "panstarrs:z", "panstarrs:y",
        "tycho:BT", "tycho:VT",
        "stromgren:u", "stromgren:v", "stromgren:b", "stromgren:y",
        "spitzer:IRAC1", "spitzer:IRAC2",
        "spitzer:IRAC3", "spitzer:IRAC4",
        "kepler:Kepler",
    ]

    _UNIT_CHOICES = ["Jy", "mJy", "uJy"]

    _CMAP_CHOICES = [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "gray", "hot", "coolwarm", "RdBu_r", "turbo",
        "afmhot", "gist_heat", "YlGnBu", "Spectral_r",
    ]

    _PROJECTION_CHOICES = [
        "native", "rect", "aitoff", "hammer", "mollweide", "lambert",
    ]

    _COORDSYS_CHOICES = ["auto", "galactic", "equatorial"]

    _NCOMP_CHOICES = ["auto", "1", "2", "3", "4", "5", "6", "7", "8"]
    _BASELINE_CHOICES = ["const", "linear"]

    # ---- dist_dm 下拉 ----
    _DM_MODEL_CHOICES = ["fixed", "ne2001", "ymw16"]
    _DIST_UNIT_CHOICES = ["pc", "kpc", "Mpc"]

    # ============================================================
    # 初始化
    # ============================================================
    def __init__(self, root):
        self.root = root
        root.title(f"天文工具集  v{VERSION}")
        root.geometry("1000x980")

        self.status_var = tk.StringVar(value="就绪")

        self._ui_queue = queue.Queue()
        self._pump_ui()

        # ---- 菜单栏 ----
        self._create_menu()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---- 内置选项卡 ----
        self.create_radec_tab()
        self.create_mag_flux_tab()
        self.create_dist_dm_tab()
        self.create_fits_skymap_tab()
        self.create_quickcube_tab()
        self.create_specfit_tab()

        # ---- 全局日志 ----
        self.create_global_log_tab()

        # ---- 底部状态栏 ----
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

        # ---- 快捷栏 ----
        self.create_quick_bar()

        # ---- 扩展入口 ----
        self.create_extensions()

    # ============================================================
    # 菜单栏
    # ============================================================
    def _create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="清空运行日志",
                              command=self._menu_clear_log)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="赤道 ↔ 银道",
                               command=lambda: self._select_tab(0))
        tools_menu.add_command(label="星等 ↔ 流量",
                               command=lambda: self._select_tab(1))
        tools_menu.add_command(label="距离 ↔ 色散量",
                               command=lambda: self._select_tab(2))
        tools_menu.add_command(label="FITS 天图",
                               command=lambda: self._select_tab(3))
        tools_menu.add_command(label="QuickCube",
                               command=lambda: self._select_tab(4))
        tools_menu.add_command(label="SpecFit",
                               command=lambda: self._select_tab(5))
        tools_menu.add_command(label="运行日志",
                               command=lambda: self._select_tab(6))
        menubar.add_cascade(label="工具", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="常用操作",
                              command=self.show_help_operations)
        help_menu.add_command(label="常用命令",
                              command=self.show_help_commands)
        help_menu.add_separator()
        help_menu.add_command(label="关于",
                              command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.config(menu=menubar)

    def _select_tab(self, index):
        try:
            self.notebook.select(index)
        except Exception:
            pass

    def _menu_clear_log(self):
        try:
            self.global_log.delete(1.0, tk.END)
            self.status_var.set("日志已清空")
        except Exception:
            pass

    # ============================================================
    # 帮助窗口
    # ============================================================
    def _open_help_window(self, title, content):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("900x700")

        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=8, pady=(8, 2))
        ttk.Label(top, text=title,
                  font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT)

        def _copy():
            win.clipboard_clear()
            win.clipboard_append(content)
            self.status_var.set("帮助内容已复制到剪贴板")

        ttk.Button(top, text="复制全部", command=_copy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="关闭", command=win.destroy).pack(side=tk.RIGHT, padx=4)

        text = scrolledtext.ScrolledText(
            win, wrap=tk.NONE,
            font=("Consolas", 10) if os.name == "nt" else ("Monospace", 10),
        )
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        text.insert("1.0", content)
        text.configure(state=tk.DISABLED)
        text.see("1.0")

        try:
            self.root.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - 900) // 2
            y = self.root.winfo_y() + 60
            win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except Exception:
            pass

        return win

    def show_help_operations(self):
        self._open_help_window("常用操作（中文手册）", HELP_OPERATIONS)

    def show_help_commands(self):
        self._open_help_window("常用命令（CLI 示例）", HELP_COMMANDS)

    def show_about(self):
        self._open_help_window("关于 天文工具集", HELP_ABOUT)

    # ============================================================
    # 扩展接口
    # ============================================================
    def register_script(self, name, filename, description=""):
        SCRIPTS[name] = filename
        if description:
            print(f"[register_script] {name} -> {filename} ({description})")
        return self

    def add_tab(self, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        return frame

    def add_script_tab(self, title, script,
                       input_file=None, positional_input=True,
                       params=(), flags=(), button_text="运行",
                       notify=False, extra_ui=None, on_complete=None,
                       log_height=12):
        return ScriptTab(self, title, script,
                         input_file=input_file,
                         positional_input=positional_input,
                         params=params, flags=flags,
                         button_text=button_text, notify=notify,
                         extra_ui=extra_ui, on_complete=on_complete,
                         log_height=log_height)

    def add_button(self, parent, text, script=None, args=None,
                   args_provider=None, log_widget=None,
                   on_complete=None, notify=False, **layout):
        if script is None:
            btn = ttk.Button(parent, text=text,
                             command=args_provider or (lambda: None))
        else:
            def _click():
                a = args_provider() if callable(args_provider) else list(args or [])
                self.run_script_async(script, a, log_widget,
                                      on_complete=on_complete, notify=notify)
            btn = ttk.Button(parent, text=text, command=_click)

        if "row" in layout:
            col = layout.pop("column", 0)
            btn.grid(row=layout.pop("row"), column=col, **layout)
        else:
            side = layout.pop("side", tk.LEFT)
            btn.pack(side=side, **layout)
        return btn

    def add_sequence_button(self, parent, text, steps,
                            log_widget=None, on_complete=None,
                            notify=False, **layout):
        def _click():
            resolved = []
            for step in steps:
                resolved.append(step() if callable(step) else step)
            self.run_sequence_async(resolved, log_widget,
                                    on_complete=on_complete, notify=notify)

        btn = ttk.Button(parent, text=text, command=_click)
        if "row" in layout:
            col = layout.pop("column", 0)
            btn.grid(row=layout.pop("row"), column=col, **layout)
        else:
            side = layout.pop("side", tk.LEFT)
            btn.pack(side=side, **layout)
        return btn

    def add_quick_button(self, text, script=None, args=None,
                         args_provider=None, log_widget=None,
                         on_complete=None, notify=False):
        return self.add_button(self.quick_bar, text, script=script,
                               args=args, args_provider=args_provider,
                               log_widget=log_widget,
                               on_complete=on_complete, notify=notify,
                               side=tk.LEFT, padx=3)

    def add_quick_sequence(self, text, steps, log_widget=None, notify=False):
        return self.add_sequence_button(self.quick_bar, text, steps,
                                        log_widget=log_widget,
                                        notify=notify, side=tk.LEFT, padx=3)

    # ============================================================
    # 异步执行
    # ============================================================
    def run_script_async(self, name, args=None, log_widget=None,
                         on_complete=None, notify=False):
        args = list(args or [])
        log = log_widget if log_widget is not None else self.global_log
        if log is not None:
            log.delete(1.0, tk.END)
        self.status_var.set(f"运行 {name} ...")

        def worker():
            code = self._exec_script(name, args, log)
            self._ui(lambda: self._finish(name, code, log,
                                          on_complete, notify))

        threading.Thread(target=worker, daemon=True).start()

    def run_sequence_async(self, steps, log_widget=None,
                           on_complete=None, notify=False):
        log = log_widget if log_widget is not None else self.global_log
        if log is not None:
            log.delete(1.0, tk.END)
        self.status_var.set("流水线运行中...")

        def worker():
            code = 0
            total = len(steps)
            for i, (name, args) in enumerate(steps, 1):
                actual = args() if callable(args) else list(args or [])
                self._append_log(log,
                                 f"\n===== [{i}/{total}] {name} =====\n")
                code = self._exec_script(name, list(actual or []), log)
                if code != 0:
                    self._append_log(log, f"[中止] {name} 返回 {code}\n")
                    break
            self._ui(lambda: self._finish("流水线", code, log,
                                          on_complete, notify))

        threading.Thread(target=worker, daemon=True).start()

    # ============================================================
    # 内部执行
    # ============================================================
    def _pump_ui(self):
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                pass
        try:
            self.root.after(40, self._pump_ui)
        except tk.TclError:
            pass

    def _ui(self, fn):
        self._ui_queue.put(fn)

    def _append_log(self, log, text):
        if log is None:
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        self._ui(lambda: (log.insert(tk.END, text), log.see(tk.END)))

    def _resolve_script(self, name):
        filename = SCRIPTS.get(name, name)
        if os.path.isabs(filename):
            return filename
        return os.path.join(SCRIPT_DIR, filename)

    def _exec_script(self, name, args, log):
        path = self._resolve_script(name)
        if not os.path.exists(path):
            self._append_log(log, f"[错误] 找不到脚本：{path}\n")
            return 127

        cmd = [sys.executable, path] + [str(a) for a in args]
        self._append_log(log, "$ " + " ".join(cmd) + "\n")

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=SCRIPT_DIR,
            )
            for line in proc.stdout:
                self._append_log(log, line)
            return proc.wait()
        except Exception as e:
            self._append_log(log, f"[异常] {e}\n")
            return -1

    def _finish(self, label, code, log, on_complete, notify):
        if code == 0:
            self.status_var.set(f"{label} 完成")
        else:
            self.status_var.set(f"{label} 出错 ({code})")

        if on_complete:
            try:
                on_complete(code)
            except Exception as e:
                self._append_log(log, f"[on_complete 异常] {e}\n")

        if notify:
            if code == 0:
                messagebox.showinfo("完成", f"{label} 运行完成！")
            else:
                messagebox.showerror("错误", f"{label} 返回状态 {code}")

    def run_script_thread(self, script_name, args, log_widget,
                          notify=True):
        self.run_script_async(script_name, args, log_widget, notify=notify)

    # ============================================================
    # 通用参数收集
    # ============================================================
    def _run_conversion(self, script_key, vars_dict, log_widget,
                        flags=None, notify=False):
        args = []
        for key, var in vars_dict.items():
            val = var.get().strip()
            if val:
                args.extend([f"--{key.replace('_', '-')}", val])
        for var, flag in (flags or []):
            if var.get():
                args.append(flag)
        self.run_script_thread(script_key, args, log_widget, notify=notify)

    # ============================================================
    # 选项卡 1：赤道 ↔ 银道
    # ============================================================
    def create_radec_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="赤道 ↔ 银道")

        params = [
            ("赤经 RA (hms 或 deg) (--ra)", "ra", "08h35m20.61125s"),
            ("赤纬 Dec (dms 或 deg) (--dec)", "dec", "-45d10m34.875s"),
            ("银经 l (deg) (--glon，反向时用)", "glon", ""),
            ("银纬 b (deg) (--glat，反向时用)", "glat", ""),
        ]
        self.radec_vars = {}
        row = 0
        for label, key, default in params:
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=var, width=40).grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.radec_vars[key] = var
            row += 1

        self.radec_decimal_in = tk.BooleanVar(value=False)
        self.radec_decimal_out = tk.BooleanVar(value=False)
        self.radec_reverse = tk.BooleanVar(value=False)

        ttk.Checkbutton(frame, text="输入为十进制度 (--decimal-in)",
                        variable=self.radec_decimal_in
                        ).grid(row=row, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(frame, text="输出为十进制度 (--decimal-out)",
                        variable=self.radec_decimal_out
                        ).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1
        ttk.Checkbutton(frame, text="反向：银道 → 赤道 (--reverse)",
                        variable=self.radec_reverse
                        ).grid(row=row, column=0, columnspan=2,
                               sticky=tk.W, padx=5)
        row += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="转换",
                   command=self.run_radec_convert
                   ).pack(side=tk.LEFT, padx=5)

        self.radec_log = scrolledtext.ScrolledText(frame, height=12, width=80)
        self.radec_log.grid(row=row + 1, column=0, columnspan=3,
                            padx=5, pady=5, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row + 1, weight=1)

    def run_radec_convert(self):
        self._run_conversion(
            'radec2glongglat', self.radec_vars, self.radec_log,
            flags=[
                (self.radec_decimal_in, "--decimal-in"),
                (self.radec_decimal_out, "--decimal-out"),
                (self.radec_reverse, "--reverse"),
            ],
            notify=False,
        )

    # ============================================================
    # 选项卡 2：星等 ↔ 流量
    # ============================================================
    def create_mag_flux_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="星等 ↔ 流量")

        self.magflux_vars = {}
        self.magflux_combo = {}
        row = 0

        def add_entry(label, key, default=""):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=v, width=32).grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.magflux_vars[key] = v
            row += 1

        def add_combo(label, key, values, default, width=30):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Combobox(frame, textvariable=v, values=values, width=width,
                         state="readonly").grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.magflux_combo[key] = v
            row += 1

        add_entry("星等 m (--mag)", "mag", "10.0")
        add_entry("流量密度 (--flux，反向时用)", "flux", "")

        add_combo("零点体系 (--system)", "system",
                  self._SYSTEM_CHOICES, "auto")
        add_combo("测光系统 (--survey)", "survey",
                  self._SURVEY_CHOICES, "auto")
        add_combo("波段/滤光片 (--band)", "band",
                  self._BAND_CHOICES, "", width=40)

        add_entry("自定义零点 (Jy) (--zero-point)", "zero_point", "")
        add_entry("有效波长 (Å) (--wavelength，留空=自动)", "wavelength", "")
        add_combo("流量单位 (--unit)", "unit", self._UNIT_CHOICES, "Jy")

        self.magflux_reverse = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="反向：流量 → 星等 (--reverse)",
                        variable=self.magflux_reverse
                        ).grid(row=row, column=0, columnspan=2,
                               sticky=tk.W, padx=5)
        row += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="转换",
                   command=self.run_mag_flux).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="列出全部波段",
                   command=self.show_band_list).pack(side=tk.LEFT, padx=5)
        row += 1

        self.magflux_log = scrolledtext.ScrolledText(frame, height=12, width=80)
        self.magflux_log.grid(row=row, column=0, columnspan=3,
                              padx=5, pady=5, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

    def run_mag_flux(self):
        args = []

        for key in ("mag", "flux", "zero_point", "wavelength"):
            val = self.magflux_vars[key].get().strip()
            if val:
                args.extend([f"--{key.replace('_', '-')}", val])

        sys_val = self.magflux_combo['system'].get()
        if sys_val and sys_val != "auto":
            args.extend(["--system", sys_val])

        survey_val = self.magflux_combo['survey'].get()
        if survey_val and survey_val != "auto":
            args.extend(["--survey", survey_val])

        band_val = self.magflux_combo['band'].get().strip()
        if band_val:
            args.extend(["--band", band_val])

        unit_val = self.magflux_combo['unit'].get()
        if unit_val:
            args.extend(["--unit", unit_val])

        reverse = self.magflux_reverse.get()
        if reverse:
            if not self.magflux_vars['flux'].get().strip():
                messagebox.showerror("错误", "反向转换需要填写「流量密度」")
                return
            args.append("--reverse")
        else:
            if not self.magflux_vars['mag'].get().strip():
                messagebox.showerror("错误", "正向转换需要填写「星等」")
                return

        self.run_script_thread('mag_flux', args, self.magflux_log,
                               notify=False)

    def show_band_list(self):
        self.run_script_thread('mag_flux', ["--list-bands"],
                               self.magflux_log, notify=False)

    # ============================================================
    # 选项卡 3：距离 ↔ 色散量（含模型下拉 + l/b 输入 + 退化确认）
    # ============================================================
    def create_dist_dm_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="距离 ↔ 色散量")

        # spec = (label, key, default) 或 (label, key, default, choices)
        params = [
            ("距离 (--distance)", "distance", "1.0"),
            ("距离单位 pc/kpc/Mpc (--dist-unit)", "dist_unit", "kpc",
             self._DIST_UNIT_CHOICES),
            ("色散量 DM (pc/cm³) (--dm，反向时用)", "dm", ""),
            ("银河电子密度模型 (--model)", "model", "fixed",
             self._DM_MODEL_CHOICES),
            ("固定电子密度 (cm⁻³) (--ne，仅 fixed)", "ne", "0.03"),
            ("银经 l (deg) (--l，ne2001/ymw16 需要)", "l", ""),
            ("银纬 b (deg) (--b，ne2001/ymw16 需要)", "b", ""),
            ("红移 (可选) (--z，配合 --cosmic)", "z", ""),
        ]
        self.distdm_vars = {}
        row = 0
        for spec in params:
            label, key, default = spec[0], spec[1], spec[2]
            choices = spec[3] if len(spec) > 3 else None

            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=default)
            if choices:
                ttk.Combobox(frame, textvariable=var, values=choices,
                             width=37, state="readonly").grid(
                    row=row, column=1, sticky=tk.W, padx=5, pady=2)
            else:
                ttk.Entry(frame, textvariable=var, width=40).grid(
                    row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.distdm_vars[key] = var
            row += 1

        self.distdm_reverse = tk.BooleanVar(value=False)
        self.distdm_cosmic = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="反向：DM → 距离 (--reverse)",
                        variable=self.distdm_reverse
                        ).grid(row=row, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(frame, text="包含宇宙学贡献 (--cosmic)",
                        variable=self.distdm_cosmic
                        ).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(
            frame,
            text="提示：ne2001 / ymw16 模型需要同时填银经 l 和银纬 b，"
                 "并确保已安装 pygedm；否则脚本会自动退化为 fixed 模型"
                 "（使用固定电子密度 --ne）。",
            foreground="gray", wraplength=760, justify=tk.LEFT,
        ).grid(row=row, column=0, columnspan=3,
               sticky=tk.W, padx=5, pady=(4, 2))
        row += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="转换",
                   command=self.run_dist_dm).pack(side=tk.LEFT, padx=5)

        self.distdm_log = scrolledtext.ScrolledText(frame, height=12, width=80)
        self.distdm_log.grid(row=row + 1, column=0, columnspan=3,
                             padx=5, pady=5, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row + 1, weight=1)

    def run_dist_dm(self):
        model = self.distdm_vars["model"].get().strip() or "fixed"
        l = self.distdm_vars["l"].get().strip()
        b = self.distdm_vars["b"].get().strip()

        # 主参数校验
        reverse = self.distdm_reverse.get()
        if reverse:
            if not self.distdm_vars["dm"].get().strip():
                messagebox.showerror("错误",
                                     "反向转换需要填写「色散量 DM」")
                return
        else:
            if not self.distdm_vars["distance"].get().strip():
                messagebox.showerror("错误",
                                     "正向转换需要填写「距离」")
                return

        # 模型前置校验：ne2001/ymw16 缺 l/b 时弹窗确认退化
        if model in ("ne2001", "ymw16") and (not l or not b):
            ok = messagebox.askyesno(
                "缺少银道坐标",
                f"模型 {model} 需要同时填写「银经 l」和「银纬 b」。\n\n"
                f"如果继续，dist_dm_convert.py 会自动退化为 "
                f"fixed 模型（使用固定电子密度 --ne）。\n\n"
                f"是否继续？"
            )
            if not ok:
                return

        # cosmic 需要 z
        if self.distdm_cosmic.get() and \
                not self.distdm_vars["z"].get().strip():
            messagebox.showerror("错误",
                                 "勾选「包含宇宙学贡献」时必须填写「红移」")
            return

        # 收集参数
        args = []
        for key in ("distance", "dist_unit", "dm", "model",
                    "ne", "l", "b", "z"):
            val = self.distdm_vars[key].get().strip()
            if val:
                args.extend([f"--{key.replace('_', '-')}", val])

        if self.distdm_reverse.get():
            args.append("--reverse")
        if self.distdm_cosmic.get():
            args.append("--cosmic")

        self.run_script_thread('dist_dm', args, self.distdm_log,
                               notify=False)

    # ============================================================
    # 选项卡 4：FITS 天图
    # ============================================================
    def create_fits_skymap_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="FITS 天图")

        self.fits_vars = {}
        self.fits_combo = {}
        row = 0

        def add_entry(label, key, default="", width=44, browse=False):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=v, width=width).grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.fits_vars[key] = v
            if browse:
                ttk.Button(frame, text="浏览...",
                           command=lambda vv=v: self.browse_file(vv)
                           ).grid(row=row, column=2, padx=5)
            row += 1

        def add_combo(label, key, values, default, width=41):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Combobox(frame, textvariable=v, values=values, width=width,
                         state="readonly").grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.fits_combo[key] = v
            row += 1

        add_entry("FITS 文件 (--input，逗号分隔多个)", "input",
                  browse=True)
        add_entry("HDU 选择 (--hdu)", "hdu", default="auto")
        add_entry("3D 切片 (--slice)", "slice", default="")
        add_entry("面板标题 (--labels，逗号分隔)", "labels", default="")

        add_combo("投影 (--projection)", "projection",
                  self._PROJECTION_CHOICES, "native")
        add_combo("坐标系 (--coordsys)", "coordsys",
                  self._COORDSYS_CHOICES, "auto")
        add_entry("中心经度 (--center-lon，留空=自动)", "center_lon", "")
        add_entry("球面最大边长像素 (--max-spherical-pixels)",
                  "max_spherical_pixels", "720")

        add_combo("Cmap (--cmap)", "cmap", self._CMAP_CHOICES, "viridis")
        add_entry("颜色范围分位 (%) (--percentile)", "percentile",
                  "0.5,99.5")
        add_entry("vmin (--vmin)", "vmin", "")
        add_entry("vmax (--vmax)", "vmax", "")
        add_entry("子图列数 (--ncols，0=自动)", "ncols", "0")
        add_entry("标题 (--title)", "title", "FITS Sky Map")
        add_entry("输出 PNG (--output)", "output", "skymap_fits.png")
        add_entry("DPI (--dpi)", "dpi", "150")

        self.fits_log_scale = tk.BooleanVar(value=False)
        self.fits_show_hdu = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="对数归一化 (--log)",
                        variable=self.fits_log_scale
                        ).grid(row=row, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(frame, text="标题带 HDU 名 (--show-hdu)",
                        variable=self.fits_show_hdu
                        ).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="绘制天图",
                   command=self.run_fits_skymap).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="列出 HDU 结构",
                   command=self.run_fits_list_hdu
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="生成模拟 2D FITS",
                   command=self.run_make_mock).pack(side=tk.LEFT, padx=5)
        row += 1

        self.fits_log = scrolledtext.ScrolledText(frame, height=10, width=80)
        self.fits_log.grid(row=row, column=0, columnspan=3,
                           padx=5, pady=5, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

    def run_fits_skymap(self):
        files = self.fits_vars["input"].get().strip()
        if not files:
            messagebox.showerror("错误", "请填写 FITS 文件路径")
            return

        args = ["--input", files]

        hdu = self.fits_vars["hdu"].get().strip()
        if hdu and hdu != "auto":
            args.extend(["--hdu", hdu])

        sl = self.fits_vars["slice"].get().strip()
        if sl:
            args.extend(["--slice", sl])

        labels = self.fits_vars["labels"].get().strip()
        if labels:
            args.extend(["--labels", labels])

        proj = self.fits_combo["projection"].get()
        if proj and proj != "native":
            args.extend(["--projection", proj])

        csys = self.fits_combo["coordsys"].get()
        if csys and csys != "auto":
            args.extend(["--coordsys", csys])

        cl = self.fits_vars["center_lon"].get().strip()
        if cl:
            args.extend(["--center-lon", cl])

        msp = self.fits_vars["max_spherical_pixels"].get().strip()
        if msp and msp != "720":
            args.extend(["--max-spherical-pixels", msp])

        for key in ("percentile", "vmin", "vmax", "title", "output", "dpi"):
            val = self.fits_vars[key].get().strip()
            if val:
                args.extend([f"--{key}", val])

        cmap = self.fits_combo["cmap"].get()
        if cmap:
            args.extend(["--cmap", cmap])

        ncols = self.fits_vars["ncols"].get().strip()
        if ncols and ncols != "0":
            args.extend(["--ncols", ncols])

        if self.fits_log_scale.get():
            args.append("--log")
        if self.fits_show_hdu.get():
            args.append("--show-hdu")

        args.append("--no-show")
        self.run_script_thread('skymap', args, self.fits_log, notify=False)

    def run_fits_list_hdu(self):
        files = self.fits_vars["input"].get().strip()
        if not files:
            messagebox.showerror("错误", "请填写 FITS 文件路径")
            return
        self.run_script_thread('skymap',
                               ["--input", files, "--list-hdu"],
                               self.fits_log, notify=False)

    def run_make_mock(self):
        args = ["--output", "mock_hi.fits"]
        out_png = self.fits_vars["output"].get().strip()
        if out_png:
            stem, _ = os.path.splitext(os.path.basename(out_png))
            if stem:
                args = ["--output", f"{stem.replace('_map', '')}.fits"]
        self.run_script_thread('make_mock', args, self.fits_log,
                               notify=False)

    # ============================================================
    # 选项卡 5：QuickCube
    # ============================================================
    def create_quickcube_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="QuickCube")

        self.qc_vars = {}
        self.qc_combo = {}
        row = 0

        def add_entry(label, key, default="", width=44, browse=False):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=v, width=width).grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.qc_vars[key] = v
            if browse:
                def _browse():
                    f = filedialog.askopenfilename(
                        filetypes=[
                            ("FITS", "*.fits *.fit *.fts *.fz"),
                            ("All files", "*.*"),
                        ])
                    if f:
                        cur = v.get().strip()
                        if cur:
                            parts = [x.strip() for x in cur.split(",") if x.strip()]
                            if f not in parts:
                                v.set(cur + "," + f)
                        else:
                            v.set(f)
                ttk.Button(frame, text="浏览...",
                           command=_browse).grid(row=row, column=2, padx=5)
            row += 1

        def add_combo(label, key, values, default, width=41):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Combobox(frame, textvariable=v, values=values, width=width,
                         state="readonly").grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.qc_combo[key] = v
            row += 1

        add_entry("FITS 文件 (--input，逗号分隔多个)", "input",
                  browse=True)
        add_entry("HDU (--hdu)", "hdu", default="auto")
        add_entry("初始切片 (--slice)", "slice", "0")
        add_entry("最大边长像素 (--max-pixels)", "max_pixels", "2048")

        add_combo("叠加模式 (--stack)", "stack",
                  ["single", "rgb"], "single")
        add_combo("Cmap (--cmap)", "cmap", self._CMAP_CHOICES, "viridis")
        add_entry("颜色范围分位 (%) (--percentile)", "percentile",
                  "0.5,99.5")
        add_entry("vmin (--vmin)", "vmin", "")
        add_entry("vmax (--vmax)", "vmax", "")

        add_entry("Mask 输出 (--mask-output)", "mask_output", "mask.fits")

        self.qc_log_scale = tk.BooleanVar(value=False)
        self.qc_no_prompt = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="对数归一化 (--log)",
                        variable=self.qc_log_scale
                        ).grid(row=row, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(frame, text="退出不提示保存 (--no-prompt-save)",
                        variable=self.qc_no_prompt
                        ).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(
            frame,
            text="提示：点击「启动交互窗口」会弹出独立 matplotlib 窗口。"
                 "键盘操作：r=标记矩形  u=撤销  s=保存 mask  c=清空  "
                 "n/p=切换切片  q=退出",
            foreground="gray", wraplength=760, justify=tk.LEFT,
        ).grid(row=row, column=0, columnspan=3,
               sticky=tk.W, padx=5, pady=(4, 2))
        row += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="启动交互窗口",
                   command=self.run_quickcube
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="仅加载测试 (--no-show)",
                   command=lambda: self.run_quickcube(no_show=True)
                   ).pack(side=tk.LEFT, padx=5)
        row += 1

        self.qc_log = scrolledtext.ScrolledText(frame, height=10, width=80)
        self.qc_log.grid(row=row, column=0, columnspan=3,
                         padx=5, pady=5, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

    def run_quickcube(self, no_show=False):
        files = self.qc_vars["input"].get().strip()
        if not files:
            messagebox.showerror("错误", "请填写 FITS 文件路径")
            return

        args = ["--input", files]

        hdu = self.qc_vars["hdu"].get().strip()
        if hdu and hdu != "auto":
            args.extend(["--hdu", hdu])

        sl = self.qc_vars["slice"].get().strip()
        if sl and sl != "0":
            args.extend(["--slice", sl])

        mp = self.qc_vars["max_pixels"].get().strip()
        if mp and mp != "2048":
            args.extend(["--max-pixels", mp])

        stack = self.qc_combo["stack"].get()
        if stack and stack != "single":
            args.extend(["--stack", stack])

        cmap = self.qc_combo["cmap"].get()
        if cmap:
            args.extend(["--cmap", cmap])

        for key in ("percentile", "vmin", "vmax"):
            val = self.qc_vars[key].get().strip()
            if val:
                args.extend([f"--{key}", val])

        mask_out = self.qc_vars["mask_output"].get().strip()
        if mask_out:
            args.extend(["--mask-output", mask_out])

        if self.qc_log_scale.get():
            args.append("--log")
        if self.qc_no_prompt.get():
            args.append("--no-prompt-save")
        if no_show:
            args.append("--no-show")

        self.run_script_thread('quickcube', args, self.qc_log,
                               notify=not no_show)

    # ============================================================
    # 选项卡 6：SpecFit
    # ============================================================
    def create_specfit_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="SpecFit")

        self.sf_vars = {}
        self.sf_combo = {}
        row = 0

        def add_entry(label, key, default="", width=44, browse=False):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=v, width=width).grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.sf_vars[key] = v
            if browse:
                ttk.Button(frame, text="浏览...",
                           command=lambda vv=v: self.browse_file(vv)
                           ).grid(row=row, column=2, padx=5)
            row += 1

        def add_combo(label, key, values, default, width=41):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0,
                                              sticky=tk.W, padx=5, pady=2)
            v = tk.StringVar(value=default)
            ttk.Combobox(frame, textvariable=v, values=values, width=width,
                         state="readonly").grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.sf_combo[key] = v
            row += 1

        add_entry("3D cube FITS (--input)", "input", browse=True)
        add_entry("HDU (--hdu)", "hdu", "auto")
        add_entry("速度范围 (--velocity-range，如 -200,200)", "velrange", "")

        add_combo("成分数 (--ncomp，auto=BIC 自适应)",
                  "ncomp", self._NCOMP_CHOICES, "1")
        add_entry("auto 最大成分数 (--max-ncomp)", "max_ncomp", "5")
        add_combo("基线 (--baseline)", "baseline",
                  self._BASELINE_CHOICES, "const")
        add_entry("S/N 阈值 (--snr-min)", "snr_min", "3")
        add_entry("并行进程 (--nproc)", "nproc", "1")

        add_entry("参数 FITS (--param-output)", "param_output",
                  "specfit_params.fits")
        add_entry("参数图前缀 (--plot-output，留空=不出 PNG)",
                  "plot_output", "")
        add_combo("Cmap (--cmap)", "cmap", self._CMAP_CHOICES, "viridis")
        add_entry("DPI (--dpi)", "dpi", "150")

        self.sf_no_show = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="自动模式不弹窗 (--no-show)",
                        variable=self.sf_no_show
                        ).grid(row=row, column=0, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(
            frame,
            text="提示：ncomp=auto 时按 BIC 在 1~max-ncomp 间自动选择成分数；"
                 "交互模式点击弹出独立窗口，点积分图看拟合。"
                 "无 cube 时可点「生成模拟 cube」自动生成一份并填入路径。"
                 "拟合完成后可点「查看参数 FITS」检查或与 truth 对比。",
            foreground="gray", wraplength=760, justify=tk.LEFT,
        ).grid(row=row, column=0, columnspan=3,
               sticky=tk.W, padx=5, pady=(4, 2))
        row += 1

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="自动批量拟合",
                   command=lambda: self.run_specfit(mode="auto")
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="交互查看",
                   command=lambda: self.run_specfit(mode="interactive")
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="查看 cube 信息",
                   command=self.run_specfit_info
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="生成模拟 cube",
                   command=self.run_make_mock_cube
                   ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="查看参数 FITS",
                   command=self.run_inspect_params
                   ).pack(side=tk.LEFT, padx=5)
        row += 1

        self.sf_log = scrolledtext.ScrolledText(frame, height=12, width=80)
        self.sf_log.grid(row=row, column=0, columnspan=3,
                         padx=5, pady=5, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

    def run_specfit(self, mode="auto"):
        f = self.sf_vars["input"].get().strip()
        if not f:
            messagebox.showerror("错误", "请填写 3D cube FITS 路径")
            return

        args = ["--input", f, "--mode", mode]

        hdu = self.sf_vars["hdu"].get().strip()
        if hdu and hdu != "auto":
            args.extend(["--hdu", hdu])

        vr = self.sf_vars["velrange"].get().strip()
        if vr:
            args.extend(["--velocity-range", vr])

        nc = self.sf_combo["ncomp"].get()
        if nc == "auto":
            args.append("--auto-ncomp")
            mn = self.sf_vars["max_ncomp"].get().strip()
            if mn:
                args.extend(["--max-ncomp", mn])
        elif nc:
            args.extend(["--ncomp", nc])

        bl = self.sf_combo["baseline"].get()
        if bl:
            args.extend(["--baseline", bl])

        for key in ("snr_min", "nproc"):
            val = self.sf_vars[key].get().strip()
            if val:
                args.extend([f"--{key.replace('_', '-')}", val])

        if mode == "auto":
            po = self.sf_vars["param_output"].get().strip()
            if po:
                args.extend(["--param-output", po])
            pl = self.sf_vars["plot_output"].get().strip()
            if pl:
                args.extend(["--plot-output", pl])
            cm = self.sf_combo["cmap"].get()
            if cm:
                args.extend(["--cmap", cm])
            dp = self.sf_vars["dpi"].get().strip()
            if dp:
                args.extend(["--dpi", dp])
            if self.sf_no_show.get():
                args.append("--no-show")

        self.run_script_thread('specfit', args, self.sf_log,
                               notify=(mode == "auto"))

    def run_specfit_info(self):
        f = self.sf_vars["input"].get().strip()
        if not f:
            messagebox.showerror("错误", "请填写 3D cube FITS 路径")
            return
        args = ["--input", f, "--info"]
        hdu = self.sf_vars["hdu"].get().strip()
        if hdu and hdu != "auto":
            args.extend(["--hdu", hdu])
        self.run_script_thread('specfit', args, self.sf_log, notify=False)

    def run_make_mock_cube(self):
        out_cube = "mock_cube.fits"
        out_mom0 = "mock_cube_mom0.fits"
        out_truth = "mock_cube_truth.fits"

        args = [
            "--output", out_cube,
            "--moment0", out_mom0,
            "--save-truth", out_truth,
        ]

        def _after(code):
            if code == 0:
                self._ui(lambda: self.sf_vars["input"].set(out_cube))

        self.run_script_async('make_mock_cube', args, self.sf_log,
                              on_complete=_after, notify=False)

    def run_inspect_params(self):
        params = self.sf_vars["param_output"].get().strip()
        if not params:
            messagebox.showerror("错误", "请先填写「参数 FITS」路径")
            return
        if not os.path.exists(params):
            messagebox.showerror("错误", f"参数 FITS 不存在：{params}")
            return

        args = ["--input", params]

        prefix = self.sf_vars["plot_output"].get().strip()
        if prefix:
            args.extend(["--plot", f"{prefix}_panel.png"])

        truth = "mock_cube_truth.fits"
        if os.path.exists(truth):
            args.extend(["--compare", truth])

        cm = self.sf_combo["cmap"].get()
        if cm:
            args.extend(["--cmap", cm])

        dp = self.sf_vars["dpi"].get().strip()
        if dp:
            args.extend(["--dpi", dp])

        self.run_script_thread('inspect_specfit', args, self.sf_log,
                               notify=False)

    # ============================================================
    # 全局日志 / 快捷栏
    # ============================================================
    def create_global_log_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="运行日志")
        self.global_log = scrolledtext.ScrolledText(frame,
                                                    height=20, width=100)
        self.global_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_quick_bar(self):
        self.quick_bar = ttk.Frame(self.root)
        self.quick_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        ttk.Label(self.quick_bar, text="快捷操作：").pack(side=tk.LEFT)

        self.add_quick_button(
            "清空日志",
            args_provider=lambda: self.global_log.delete(1.0, tk.END)
        )

        self.add_quick_button(
            "示例：RA/Dec → l/b",
            script="radec2glongglat",
            args_provider=lambda: [
                "--ra", "08h35m20.61125s",
                "--dec", "-45d10m34.875s",
                "--decimal-out",
            ],
            log_widget=self.global_log,
            notify=True,
        )

        self.add_quick_button(
            "示例：Vega V=10 → 流量",
            script="mag_flux",
            args_provider=lambda: [
                "--mag", "10", "--band", "johnson:V",
            ],
            log_widget=self.global_log,
            notify=True,
        )

        self.add_quick_sequence(
            "模拟 FITS → Aitoff 天图",
            steps=[
                ("make_mock", lambda: [
                    "--output", "mock_hi.fits",
                    "--n-sources", "300",
                    "--noise", "0.05",
                ]),
                ("skymap", lambda: [
                    "--input", "mock_hi.fits",
                    "--projection", "aitoff",
                    "--coordsys", "galactic",
                    "--log", "--cmap", "viridis",
                    "--percentile", "0.5,99.5",
                    "--output", "mock_hi_aitoff.png",
                    "--no-show",
                ]),
            ],
            log_widget=self.global_log,
            notify=True,
        )

        self.add_quick_sequence(
            "模拟 FITS → QuickCube",
            steps=[
                ("make_mock", lambda: [
                    "--output", "mock_hi.fits",
                    "--n-sources", "300",
                    "--noise", "0.05",
                ]),
                ("quickcube", lambda: [
                    "--input", "mock_hi.fits",
                    "--mask-output", "mock_hi_mask.fits",
                    "--no-prompt-save",
                ]),
            ],
            log_widget=self.global_log,
            notify=False,
        )

        self.add_quick_sequence(
            "模拟 cube → SpecFit → 参数检查",
            steps=[
                ("make_mock_cube", lambda: [
                    "--output", "mock_cube.fits",
                    "--moment0", "mock_cube_mom0.fits",
                    "--save-truth", "mock_cube_truth.fits",
                    "--ncomp-max", "3",
                ]),
                ("specfit", lambda: [
                    "--input", "mock_cube.fits",
                    "--auto-ncomp", "--max-ncomp", "3",
                    "--snr-min", "3", "--nproc", "2",
                    "--param-output", "mock_cube_params.fits",
                    "--plot-output", "mock_cube_maps",
                    "--no-show",
                ]),
                ("inspect_specfit", lambda: [
                    "--input", "mock_cube_params.fits",
                    "--plot", "mock_cube_maps_panel.png",
                    "--compare", "mock_cube_truth.fits",
                    "--cmap", "viridis",
                ]),
            ],
            log_widget=self.global_log,
            notify=True,
        )

    # ============================================================
    # 扩展入口
    # ============================================================
    def create_extensions(self):
        """在这里注册自定义脚本 / 选项卡 / 快捷按钮。

        示例（取消注释即可用）：

        # A. 声明式新增 "红移 ↔ 距离" 选项卡
        # self.register_script("z2dist", "redshift_distance.py")
        # self.add_script_tab(
        #     title="红移 ↔ 距离",
        #     script="z2dist",
        #     params=[
        #         P("红移 z (--z)", "z", "0.1"),
        #         P("H0 (--h0)", "h0", "70.0"),
        #     ],
        #     flags=[F("反向 (--reverse)", "reverse")],
        # )

        # B. 快捷栏流水线：RA/Dec → l/b → DM
        # self.add_quick_sequence(
        #     "一键：RA/Dec → l/b → DM",
        #     steps=[
        #         ("radec2glongglat", lambda: [
        #             "--ra", "08h35m20.61125s",
        #             "--dec", "-45d10m34.875s",
        #         ]),
        #         ("dist_dm", lambda: [
        #             "--distance", "1.0", "--dist-unit", "kpc",
        #         ]),
        #     ],
        #     log_widget=self.global_log,
        #     notify=True,
        # )
        """
        pass

    # ============================================================
    # 辅助
    # ============================================================
    def browse_file(self, var, save=False, mode=None,
                    filetypes=None, defaultextension=".txt"):
        if mode is None:
            mode = "save" if save else "open"

        file = None
        if mode == "save":
            file = filedialog.asksaveasfilename(
                defaultextension=defaultextension,
                filetypes=filetypes or [("All files", "*.*")])
        elif mode == "dir":
            file = filedialog.askdirectory()
        else:
            file = filedialog.askopenfilename(
                filetypes=filetypes or [
                    ("FITS files", "*.fits *.fit *.fts *.fz *.fits.gz"),
                    ("Python scripts", "*.py"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ])
        if file:
            var.set(file)

    def _copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("已复制到剪贴板")


# ================== 主程序 ==================
if __name__ == "__main__":
    root = tk.Tk()
    app = AstroToolsGUI(root)
    root.mainloop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quickcube.py
============
快速查看 2D 图像 / 3D cube 切片，交互标记坏点 / 坏区域

功能
----
- 支持单文件 2D 图像、3D cube（用 n/p 键切切片）
- 多文件 RGB 叠加显示（--stack rgb，需要恰好 3 个文件）
- 鼠标滚轮缩放，matplotlib 工具栏拖动
- 交互标记坏点：按 r 进入标记模式，拖拽画矩形
- 实时显示坏像素比例
- 保存 mask 为 FITS（uint8，1=坏）

键盘操作
--------
  r      切换标记模式（进入后拖拽鼠标画矩形）
  u      撤销最后一次标记
  s      保存 mask 到 --mask-output
  c      清空所有标记
  n/p    3D cube 的下一/上一切片
  g      切换经纬网格
  h      打印帮助
  q/Esc  退出

依赖
----
numpy, matplotlib, astropy

示例
----
  python quickcube.py --input cube.fits --slice 5
  python quickcube.py --input img.fits --mask-output mask.fits --log
  python quickcube.py --inputs a.fits,b.fits,c.fits --stack rgb
  python quickcube.py --input img.fits --no-show            # 测试加载
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
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import Rectangle
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


# ==================== 数据加载 ====================
def load_fits_image(path, hdu_spec=None, slice_idx=0, max_pixels=2048):
    if not os.path.exists(path):
        raise ValueError(f"文件不存在：{path}")

    with fits.open(path) as hdul:
        # ---- 选 HDU ----
        if hdu_spec is None or str(hdu_spec).strip().lower() == "auto":
            idx = None
            for i, h in enumerate(hdul):
                if h.data is not None and h.data.ndim in (2, 3):
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"{path}: 找不到 2D/3D 图像 HDU")
        else:
            try:
                idx = int(hdu_spec)
            except ValueError:
                raise ValueError(f"{path}: 无效 --hdu={hdu_spec!r}")

        hdu = hdul[idx]
        if hdu.data is None:
            raise ValueError(f"{path}: HDU[{idx}] 无数据")
        data = hdu.data
        if data.ndim < 2:
            raise ValueError(f"{path}: HDU[{idx}] 是 {data.ndim}D")

        header = hdu.header.copy()
        if header.get("PIXTYPE", "").upper() == "HEALPIX":
            raise ValueError(f"{path}: HEALPix 格式不支持直接查看")

        # ---- 降采样 ----
        if max_pixels and max_pixels > 0:
            ny, nx = data.shape[-2:]
            stride = max(1, max(ny, nx) // max_pixels)
            if stride > 1:
                if data.ndim == 2:
                    data = data[::stride, ::stride]
                else:
                    data = data[..., ::stride, ::stride]

        # ---- cube 切片 ----
        if data.ndim == 3:
            n_slices = data.shape[0]
            if not (0 <= slice_idx < n_slices):
                slice_idx = 0
            full_cube = data
            data_2d = np.asarray(data[slice_idx], dtype=float)
        elif data.ndim == 2:
            full_cube = None
            n_slices = 1
            slice_idx = 0
            data_2d = np.asarray(data, dtype=float)
        else:
            raise ValueError(f"{path}: HDU[{idx}] 是 {data.ndim}D，暂不支持")

        # ---- WCS ----
        wcs = None
        try:
            w = WCS(header)
            if w.has_celestial:
                wcs = w.celestial
        except Exception:
            pass

        unit = header.get("BUNIT", "").strip()
        return dict(path=path, data=data_2d, cube=full_cube,
                    n_slices=n_slices, slice_idx=slice_idx,
                    header=header, wcs=wcs, unit=unit, hdu_index=idx)


# ==================== 归一化 ====================
def compute_norm(data, log=False, percentile=(0.5, 99.5),
                 vmin=None, vmax=None):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("数据全 NaN，无法归一化")

    v_lo = float(np.percentile(finite, percentile[0]))
    v_hi = float(np.percentile(finite, percentile[1]))
    if vmin is not None:
        v_lo = float(vmin)
    if vmax is not None:
        v_hi = float(vmax)

    if log:
        pos = finite[finite > 0]
        if pos.size == 0:
            raise ValueError("对数拉伸需要正数据")
        if v_lo <= 0:
            v_lo = float(np.percentile(pos, 0.5))
        if v_hi <= v_lo:
            v_hi = float(pos.max())
        if v_hi <= v_lo:
            raise ValueError("无法确定对数拉伸范围")
        return colors.LogNorm(vmin=v_lo, vmax=v_hi)

    if v_hi <= v_lo:
        raise ValueError(f"非法范围 {v_lo}~{v_hi}")
    return colors.Normalize(vmin=v_lo, vmax=v_hi)


# ==================== RGB 合成 ====================
def make_rgb(arrays):
    if len(arrays) != 3:
        raise ValueError("RGB 合成需要恰好 3 个文件")
    out = []
    for a in arrays:
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            out.append(np.zeros_like(a))
            continue
        lo = float(np.percentile(finite, 0.5))
        hi = float(np.percentile(finite, 99.5))
        if hi <= lo:
            hi = lo + 1e-10
        norm = np.clip((a - lo) / (hi - lo), 0, 1)
        norm[~np.isfinite(norm)] = 0
        out.append(norm)
    return np.dstack(out)


# ==================== 交互 viewer ====================
class CubeViewer:
    def __init__(self, items, args):
        self.items = items
        self.args = args
        self.current = 0

        self.mark_mode = False
        self._drag_start = None
        self._temp_rect = None
        self.rect_patches = []
        self.mask = None

        self._setup_display()
        self._build_figure()
        self._bind_events()
        self._update_status()

    # ---------------- 显示数据准备 ----------------
    def _parse_pct(self):
        s = self.args.percentile or "0.5,99.5"
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            return 0.5, 99.5
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            return 0.5, 99.5
        if not (0 <= lo < hi <= 100):
            return 0.5, 99.5
        return lo, hi

    def _setup_display(self):
        if self.args.stack == "rgb" and len(self.items) == 3:
            arrays = [it["data"] for it in self.items]
            self.rgb = make_rgb(arrays)
            self.display_data = None
            self.base_shape = self.rgb.shape[:2]
            self.unit_label = "RGB"
        else:
            it = self.items[self.current]
            self.rgb = None
            self.display_data = it["data"]
            self.base_shape = it["data"].shape
            self.unit_label = it.get("unit") or "value"

        self.mask = np.zeros(self.base_shape, dtype=bool)

    # ---------------- 构建窗口 ----------------
    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        try:
            self.fig.canvas.manager.set_window_title("QuickCube")
        except Exception:
            pass

        if self.rgb is not None:
            self.im = self.ax.imshow(self.rgb, origin="lower",
                                     interpolation="nearest")
        else:
            norm = compute_norm(self.display_data, log=self.args.log,
                                percentile=self._parse_pct(),
                                vmin=self.args.vmin, vmax=self.args.vmax)
            self.im = self.ax.imshow(self.display_data, origin="lower",
                                     cmap=self.args.cmap, norm=norm,
                                     interpolation="nearest")
            self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.02,
                              label=self.unit_label)

        self.ax.set_title(self._title_text(), fontsize=10)
        self.ax.set_xlabel("x [pix]", fontsize=9)
        self.ax.set_ylabel("y [pix]", fontsize=9)

        self.status_text = self.fig.text(
            0.02, 0.005, "",
            fontsize=8, color="gray",
            family="monospace",
        )
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # ---------------- 事件绑定 ----------------
    def _bind_events(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    # ---------------- 鼠标 ----------------
    def _on_press(self, event):
        if not self.mark_mode:
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._drag_start = (event.xdata, event.ydata)
        if self._temp_rect is not None:
            try:
                self._temp_rect.remove()
            except Exception:
                pass
        self._temp_rect = Rectangle(
            (event.xdata, event.ydata), 0, 0,
            linewidth=1, edgecolor="yellow",
            facecolor="yellow", alpha=0.3,
        )
        self.ax.add_patch(self._temp_rect)
        self.fig.canvas.draw_idle()

    def _on_motion(self, event):
        if not self.mark_mode or self._drag_start is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.xdata, event.ydata
        xa, xb = min(x0, x1), max(x0, x1)
        ya, yb = min(y0, y1), max(y0, y1)
        if self._temp_rect is not None:
            self._temp_rect.set_xy((xa, ya))
            self._temp_rect.set_width(xb - xa)
            self._temp_rect.set_height(yb - ya)
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self.mark_mode or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None

        if self._temp_rect is not None:
            try:
                self._temp_rect.remove()
            except Exception:
                pass
            self._temp_rect = None

        if event.xdata is None or event.ydata is None:
            self.fig.canvas.draw_idle()
            return

        x1, y1 = event.xdata, event.ydata
        ny, nx = self.base_shape
        col0 = max(0, min(nx, int(np.floor(min(x0, x1)))))
        col1 = max(0, min(nx, int(np.ceil(max(x0, x1)))))
        row0 = max(0, min(ny, int(np.floor(min(y0, y1)))))
        row1 = max(0, min(ny, int(np.ceil(max(y0, y1)))))

        if row1 <= row0 or col1 <= col0:
            self.fig.canvas.draw_idle()
            return

        self.mask[row0:row1, col0:col1] = True
        rect = Rectangle(
            (col0, row0), col1 - col0, row1 - row0,
            linewidth=1, edgecolor="red", facecolor="red", alpha=0.35,
        )
        self.ax.add_patch(rect)
        self.rect_patches.append(rect)
        self.fig.canvas.draw_idle()
        self._update_status()

    # ---------------- 键盘 ----------------
    def _on_key(self, event):
        k = event.key
        if k == "r":
            self._toggle_mark_mode()
        elif k == "u":
            self._undo()
        elif k == "s":
            self._save_mask()
        elif k == "c":
            self._clear_mask()
        elif k == "n":
            self._next_slice()
        elif k == "p":
            self._prev_slice()
        elif k == "g":
            self._toggle_grid()
        elif k == "h":
            self._print_help()
        elif k in ("q", "escape"):
            self._quit()

    def _toggle_mark_mode(self):
        self.mark_mode = not self.mark_mode
        if self.mark_mode:
            self.ax.set_title(self._title_text() + "  [MARK]",
                              fontsize=10, color="red")
            self.ax.set_facecolor("#ffe5e5")
        else:
            self.ax.set_title(self._title_text(), fontsize=10, color="black")
            self.ax.set_facecolor("white")
        self.fig.canvas.draw_idle()

    def _toggle_grid(self):
        self.ax.grid(not self.ax._axisbelow or not any(
            getattr(g, "get_visible", lambda: True)() for g in self.ax.get_xgridlines()
        ) if hasattr(self.ax, "_axisbelow") else True,
            alpha=0.3, linestyle="--")
        self.fig.canvas.draw_idle()

    def _undo(self):
        if not self.rect_patches:
            return
        p = self.rect_patches.pop()
        try:
            p.remove()
        except Exception:
            pass
        self._rebuild_mask_from_patches()
        self.fig.canvas.draw_idle()
        self._update_status()

    def _rebuild_mask_from_patches(self):
        self.mask[:] = False
        ny, nx = self.base_shape
        for p in self.rect_patches:
            x, y = p.get_xy()
            w, h = p.get_width(), p.get_height()
            col0 = max(0, min(nx, int(x)))
            col1 = max(0, min(nx, int(x + w)))
            row0 = max(0, min(ny, int(y)))
            row1 = max(0, min(ny, int(y + h)))
            if row1 > row0 and col1 > col0:
                self.mask[row0:row1, col0:col1] = True

    def _clear_mask(self):
        for p in self.rect_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.rect_patches = []
        self.mask[:] = False
        self.fig.canvas.draw_idle()
        self._update_status()

    def _save_mask(self):
        out = self.args.mask_output
        if not out:
            print("[警告] 未指定 --mask-output，无法保存 mask")
            return
        header = None
        if self.rgb is None:
            header = self.items[self.current]["header"]
        fits.writeto(out, self.mask.astype(np.uint8), header, overwrite=True)
        n_bad = int(self.mask.sum())
        print(f"[保存] mask -> {out}  (坏像素数 = {n_bad})")

    def _next_slice(self):
        self._load_slice(+1)

    def _prev_slice(self):
        self._load_slice(-1)

    def _load_slice(self, delta):
        if self.rgb is not None:
            return
        it = self.items[self.current]
        if it["cube"] is None or it["n_slices"] <= 1:
            return
        n = it["n_slices"]
        new = (it["slice_idx"] + delta) % n
        it["slice_idx"] = new
        it["data"] = np.asarray(it["cube"][new], dtype=float)
        self.display_data = it["data"]
        norm = compute_norm(self.display_data, log=self.args.log,
                            percentile=self._parse_pct(),
                            vmin=self.args.vmin, vmax=self.args.vmax)
        self.im.set_data(self.display_data)
        self.im.set_norm(norm)
        self.im.set_clim(norm.vmin, norm.vmax)
        self.ax.set_title(self._title_text(), fontsize=10)
        self.fig.canvas.draw_idle()
        self._update_status()

    def _title_text(self):
        if self.rgb is not None:
            names = ", ".join(os.path.basename(it["path"]) for it in self.items)
            return f"RGB: {names}"
        it = self.items[self.current]
        base = os.path.basename(it["path"])
        if it["cube"] is not None and it["n_slices"] > 1:
            return f"{base}  slice {it['slice_idx'] + 1}/{it['n_slices']}"
        return base

    def _update_status(self):
        n_bad = int(self.mask.sum())
        total = self.mask.size
        pct = 100.0 * n_bad / total if total > 0 else 0.0
        mark_flag = "MARK-ON" if self.mark_mode else "view"
        self.status_text.set_text(
            f"[{mark_flag}]  bad: {n_bad}/{total} ({pct:.2f}%)  "
            f"blocks: {len(self.rect_patches)}  |  "
            f"r=mark  u=undo  s=save  c=clear  n/p=slice  g=grid  q=quit"
        )
        self.fig.canvas.draw_idle()

    def _print_help(self):
        print("""
QuickCube 键盘操作
------------------
  r        切换标记模式（进入后拖拽鼠标画矩形）
  u        撤销最后一次标记
  s        保存 mask 到 --mask-output
  c        清空所有标记
  n / p    下一个 / 上一个切片（3D cube）
  g        切换网格
  h        打印此帮助
  q / Esc  退出
""")

    def _on_scroll(self, event):
        """以鼠标位置为中心的滚轮缩放。"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 1.15 if event.button == "up" else 1.0 / 1.15
        xd, yd = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim((xd + (xlim[0] - xd) * scale,
                          xd + (xlim[1] - xd) * scale))
        self.ax.set_ylim((yd + (ylim[0] - yd) * scale,
                          yd + (ylim[1] - yd) * scale))
        self.fig.canvas.draw_idle()

    def _quit(self):
        if self.mask.any() and not self.args.no_prompt_save:
            try:
                ans = input("有未保存标记，是否保存？[y/N] ").strip().lower()
            except EOFError:
                ans = ""
            if ans == "y":
                self._save_mask()
        plt.close(self.fig)

    def _on_close(self, event):
        # 窗口被直接关闭（X 按钮），做个提示
        if self.mask.any() and not self.args.no_prompt_save:
            print("[提示] 窗口关闭但标记未保存"
                  f"（共 {int(self.mask.sum())} 个坏像素）")

    def show(self):
        try:
            plt.show()
        except Exception as e:
            print(f"[警告] 无法打开交互窗口：{e}", file=sys.stderr)


# ==================== CLI ====================
def build_parser():
    p = argparse.ArgumentParser(
        description="QuickCube：快速查看 2D 图像 / 3D cube，交互标记坏点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""键盘操作：
  r=标记矩形  u=撤销  s=保存mask  c=清空  n/p=切切片  q=退出

示例：
  python quickcube.py --input cube.fits --slice 5
  python quickcube.py --inputs a.fits,b.fits,c.fits --stack rgb
  python quickcube.py --input img.fits --mask-output mask.fits --log
  python quickcube.py --input img.fits --no-show      # 只测试加载
""")

    src = p.add_argument_group("数据源")
    src.add_argument("--input", default=None,
                     help="FITS 路径（支持逗号分隔多个）")
    src.add_argument("--inputs", default=None,
                     help="--input 的别名，可与 --input 组合")
    src.add_argument("--hdu", default=None,
                     help="HDU 索引（数字，默认 auto）")
    src.add_argument("--slice", type=int, default=0, dest="slice_idx",
                     help="3D cube 初始切片索引（默认 0）")
    src.add_argument("--max-pixels", type=int, default=2048,
                     help="超过该尺寸时降采样（默认 2048）")

    disp = p.add_argument_group("显示")
    disp.add_argument("--stack", choices=["single", "rgb"], default="single",
                      help="single=单图；rgb=3 个文件 RGB 合成")
    disp.add_argument("--cmap", default="viridis")
    disp.add_argument("--log", action="store_true",
                      help="对数归一化（单图模式）")
    disp.add_argument("--percentile", default="0.5,99.5",
                      help="颜色范围分位数（默认 0.5,99.5）")
    disp.add_argument("--vmin", type=float, default=None)
    disp.add_argument("--vmax", type=float, default=None)

    out = p.add_argument_group("输出")
    out.add_argument("--mask-output", default=None,
                     help="mask 保存路径（FITS，uint8，1=坏）")
    out.add_argument("--no-show", action="store_true",
                     help="不打开交互窗口，只打印信息")
    out.add_argument("--no-prompt-save", action="store_true",
                     help="退出时即使有未保存标记也不提示")
    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)

    files = []
    for s in (args.input, args.inputs):
        if s:
            files.extend([x.strip() for x in s.split(",") if x.strip()])
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]
    if not files:
        print("[错误] 请用 --input 或 --inputs 指定 FITS", file=sys.stderr)
        return 1

    try:
        items = [load_fits_image(f, args.hdu, args.slice_idx, args.max_pixels)
                 for f in files]

        if args.stack == "rgb":
            if len(items) != 3:
                raise ValueError("--stack rgb 需要恰好 3 个文件")

        shapes = {it["data"].shape for it in items}
        if len(shapes) > 1:
            raise ValueError(f"输入图像尺寸不一致：{shapes}")

        if args.no_show:
            print(f"[no-show] {len(items)} 个文件，尺寸 {list(shapes)[0]}")
            for it in items:
                print(f"  {os.path.basename(it['path'])}  "
                      f"HDU[{it['hdu_index']}]  "
                      f"slice={it['slice_idx']}/{it['n_slices']}")
            return 0

        CubeViewer(items, args).show()
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

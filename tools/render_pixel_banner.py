#!/usr/bin/env python3
"""
Reference-Conditioned True Pixel-Art Asset & Scene Renderer for Contour (v2 - Clean Masks & Sky)
================================================================================================
Parses `assets/banner.jpg` (1376x768) into its semantic layers and foreground objects,
eliminates AI "mixel" grid drift and JPEG compression noise, reconstructs occluded
backgrounds behind props, and renders each asset + the complete scene at native
480x270 VGA resolution using a unified 64-color master palette and ordered 4x4 Bayer
dithering.
"""

from __future__ import annotations
import ctypes
import math
import os
import random
import struct
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

RGBA = Tuple[int, int, int, int]
RGB = Tuple[int, int, int]
Point = Tuple[float, float]

NATIVE_W = 480
NATIVE_H = 270
FG_SCALE = 2
FG_W = NATIVE_W * FG_SCALE  # 960
FG_H = NATIVE_H * FG_SCALE  # 540

# Active interior of `assets/banner.jpg` (1376x768) excluding the thin outer frame
CROP_X0 = 6
CROP_Y0 = 4
CROP_W = 1364
CROP_H = 761

BAYER_4X4 = (
    (0,  8,  2, 10),
    (12, 4, 14,  6),
    (3, 11,  1,  9),
    (15, 7, 13,  5),
)


# ==============================================================================
# PURE-PYTHON RGBA CANVAS & PNG ENCODER
# ==============================================================================
class Canvas:
    def __init__(self, width: int = NATIVE_W, height: int = NATIVE_H) -> None:
        self.w = width
        self.h = height
        self.buf: bytearray = bytearray(width * height * 4)

    def clone(self) -> "Canvas":
        c = Canvas(self.w, self.h)
        c.buf[:] = self.buf
        return c


    def set_px(self, x: int, y: int, color: RGBA) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            idx = (y * self.w + x) * 4
            self.buf[idx] = color[0]
            self.buf[idx + 1] = color[1]
            self.buf[idx + 2] = color[2]
            self.buf[idx + 3] = color[3]

    def get_px(self, x: int, y: int) -> RGBA:
        if 0 <= x < self.w and 0 <= y < self.h:
            idx = (y * self.w + x) * 4
            return (self.buf[idx], self.buf[idx + 1], self.buf[idx + 2], self.buf[idx + 3])
        return (0, 0, 0, 0)

    def blit(self, src: "Canvas", dx: int = 0, dy: int = 0) -> None:
        sw, sh = src.w, src.h
        sbuf = src.buf
        for sy in range(sh):
            ty = sy + dy
            if not (0 <= ty < self.h):
                continue
            for sx in range(sw):
                tx = sx + dx
                if not (0 <= tx < self.w):
                    continue
                sidx = (sy * sw + sx) * 4
                sa = sbuf[sidx + 3]
                if sa == 0:
                    continue
                tidx = (ty * self.w + tx) * 4
                if sa == 255 or self.buf[tidx + 3] == 0:
                    self.buf[tidx:tidx + 4] = sbuf[sidx:sidx + 4]
                else:
                    inv = 255 - sa
                    self.buf[tidx] = (sbuf[sidx] * sa + self.buf[tidx] * inv) // 255
                    self.buf[tidx + 1] = (sbuf[sidx + 1] * sa + self.buf[tidx + 1] * inv) // 255
                    self.buf[tidx + 2] = (sbuf[sidx + 2] * sa + self.buf[tidx + 2] * inv) // 255
                    self.buf[tidx + 3] = min(255, sa + (self.buf[tidx + 3] * inv) // 255)

    def line(self, x0: float, y0: float, x1: float, y1: float, color: RGBA, thickness: int = 1) -> None:
        ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        dx = abs(ix1 - ix0)
        dy = -abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx + dy
        r = (thickness - 1) / 2.0
        r2 = r * r + 0.35
        ir = int(math.ceil(r))
        while True:
            if thickness <= 1:
                self.set_px(ix0, iy0, color)
            else:
                for oy in range(-ir, ir + 1):
                    for ox in range(-ir, ir + 1):
                        if ox * ox + oy * oy <= r2:
                            self.set_px(ix0 + ox, iy0 + oy, color)
            if ix0 == ix1 and iy0 == iy1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                ix0 += sx
            if e2 <= dx:
                err += dx
                iy0 += sy

    def polyline(self, pts: Sequence[Point], color: RGBA, thickness: int = 1, closed: bool = False) -> None:
        for i in range(len(pts) - 1):
            self.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color, thickness)
        if closed and len(pts) > 2:
            self.line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], color, thickness)

    def fill_poly(self, pts: Sequence[Point], color_or_fn) -> None:
        if len(pts) < 3:
            return
        min_y = max(0, int(math.floor(min(p[1] for p in pts))))
        max_y = min(self.h - 1, int(math.ceil(max(p[1] for p in pts))))
        n = len(pts)
        is_callable = callable(color_or_fn)
        for y in range(min_y, max_y + 1):
            yc = y + 0.5
            xs: List[float] = []
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                if (y1 <= yc < y2) or (y2 <= yc < y1):
                    t = (yc - y1) / (y2 - y1)
                    xs.append(x1 + t * (x2 - x1))
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                x_start = max(0, int(math.floor(xs[k] + 0.5)))
                x_end = min(self.w - 1, int(math.floor(xs[k + 1] - 0.5)))
                for x in range(x_start, x_end + 1):
                    c = color_or_fn(x, y) if is_callable else color_or_fn
                    if c is not None:
                        self.set_px(x, y, c)

    def crop_non_empty(self, pad: int = 2) -> "Canvas":
        min_x, min_y = self.w, self.h
        max_x, max_y = -1, -1
        for y in range(self.h):
            row_off = y * self.w * 4
            for x in range(self.w):
                if self.buf[row_off + x * 4 + 3] > 0:
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
        if max_x < min_x:
            return Canvas(1, 1)
        min_x = max(0, min_x - pad)
        min_y = max(0, min_y - pad)
        max_x = min(self.w - 1, max_x + pad)
        max_y = min(self.h - 1, max_y + pad)
        cw = max_x - min_x + 1
        ch = max_y - min_y + 1
        out = Canvas(cw, ch)
        for y in range(ch):
            for x in range(cw):
                out.set_px(x, y, self.get_px(min_x + x, min_y + y))
        return out

    def upscale_nearest(self, scale: int) -> "Canvas":
        out = Canvas(self.w * scale, self.h * scale)
        obuf = out.buf
        sbuf = self.buf
        sw, sh = self.w, self.h
        ow = out.w
        for y in range(sh):
            for x in range(sw):
                sidx = (y * sw + x) * 4
                px = sbuf[sidx:sidx + 4]
                for sy in range(scale):
                    ty = y * scale + sy
                    row_base = (ty * ow + x * scale) * 4
                    for sx in range(scale):
                        tidx = row_base + sx * 4
                        obuf[tidx:tidx + 4] = px
        return out

    def save_png(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        raw_rows = bytearray()
        stride = self.w * 4
        for y in range(self.h):
            raw_rows.append(0)
            start = y * stride
            raw_rows.extend(self.buf[start:start + stride])

        def chunk(tag: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", self.w, self.h, 8, 6, 0, 0, 0)
        idat = zlib.compress(bytes(raw_rows), level=9)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(png)


# ==============================================================================
# REFERENCE IMAGE DECODER & EDGE-LOCKED DE-MIXELING SAMPLER
# ==============================================================================
def load_reference_jpeg(path: str) -> Tuple[int, int, int, bytes]:
    gdk = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgdk_pixbuf-2.0.so.0")
    gdk.gdk_pixbuf_new_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    gdk.gdk_pixbuf_new_from_file.restype = ctypes.c_void_p
    gdk.gdk_pixbuf_get_width.argtypes = [ctypes.c_void_p]
    gdk.gdk_pixbuf_get_width.restype = ctypes.c_int
    gdk.gdk_pixbuf_get_height.argtypes = [ctypes.c_void_p]
    gdk.gdk_pixbuf_get_height.restype = ctypes.c_int
    gdk.gdk_pixbuf_get_rowstride.argtypes = [ctypes.c_void_p]
    gdk.gdk_pixbuf_get_rowstride.restype = ctypes.c_int
    gdk.gdk_pixbuf_get_pixels.argtypes = [ctypes.c_void_p]
    gdk.gdk_pixbuf_get_pixels.restype = ctypes.POINTER(ctypes.c_ubyte)

    err = ctypes.c_void_p()
    pb = gdk.gdk_pixbuf_new_from_file(path.encode("utf-8"), ctypes.byref(err))
    if not pb:
        raise RuntimeError(f"Failed to decode {path}")
    w = gdk.gdk_pixbuf_get_width(pb)
    h = gdk.gdk_pixbuf_get_height(pb)
    rs = gdk.gdk_pixbuf_get_rowstride(pb)
    raw = bytes(ctypes.cast(gdk.gdk_pixbuf_get_pixels(pb), ctypes.POINTER(ctypes.c_ubyte * (h * rs))).contents)
    return w, h, rs, raw


def fbm_hash(x: int, y: int, seed: int = 1337) -> float:
    n = (x * 374761393 + y * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65536.0


def value_noise(x: float, y: float, scale: float = 8.0, seed: int = 1337) -> float:
    gx = x / scale
    gy = y / scale
    ix = int(math.floor(gx))
    iy = int(math.floor(gy))
    fx = gx - ix
    fy = gy - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    v00 = fbm_hash(ix, iy, seed)
    v10 = fbm_hash(ix + 1, iy, seed)
    v01 = fbm_hash(ix, iy + 1, seed)
    v11 = fbm_hash(ix + 1, iy + 1, seed)
    return (v00 * (1 - ux) + v10 * ux) * (1 - uy) + (v01 * (1 - ux) + v11 * ux) * uy


def bayer_pick(x: int, y: int, ramp: Sequence[RGBA], t: float) -> RGBA:
    if t <= 0.0:
        return ramp[0]
    if t >= 1.0:
        return ramp[-1]
    scaled = t * (len(ramp) - 1)
    idx = int(math.floor(scaled))
    frac = scaled - idx
    if idx >= len(ramp) - 1:
        return ramp[-1]
    threshold = (BAYER_4X4[y & 3][x & 3] + 0.5) / 16.0
    return ramp[idx + 1] if frac > threshold else ramp[idx]


def rgb_lum(c: RGB) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def rgb_dist2(a: RGB, b: RGB) -> float:
    dr = a[0] - b[0]
    dg = a[1] - b[1]
    db = a[2] - b[2]
    return 2.2 * dr * dr + 4.0 * dg * dg + 1.8 * db * db


def sample_demixeled_grid(
    w: int, h: int, rs: int, raw: bytes, out_w: int = NATIVE_W, out_h: int = NATIVE_H
) -> List[List[RGB]]:
    grid: List[List[RGB]] = [[(0, 0, 0) for _ in range(out_w)] for _ in range(out_h)]
    cell_rx = max(0.75, 0.55 * CROP_W / float(out_w))
    cell_ry = max(0.75, 0.55 * CROP_H / float(out_h))
    for gy in range(out_h):
        yc = CROP_Y0 + (gy + 0.5) * CROP_H / float(out_h)
        iy_center = max(0, min(h - 1, int(round(yc))))
        y0 = max(CROP_Y0, int(math.floor(yc - cell_ry)))
        y1 = min(CROP_Y0 + CROP_H - 1, int(math.ceil(yc + cell_ry)))
        for gx in range(out_w):
            xc = CROP_X0 + (gx + 0.5) * CROP_W / float(out_w)
            ix_center = max(0, min(w - 1, int(round(xc))))
            x0 = max(CROP_X0, int(math.floor(xc - cell_rx)))
            x1 = min(CROP_X0 + CROP_W - 1, int(math.ceil(xc + cell_rx)))

            c_idx = iy_center * rs + ix_center * 3
            center_rgb: RGB = (raw[c_idx], raw[c_idx + 1], raw[c_idx + 2])
            center_l = rgb_lum(center_rgb)

            samples: List[Tuple[RGB, float]] = []
            min_l, max_l = 999.0, -1.0
            darkest_rgb = center_rgb
            brightest_rgb = center_rgb
            for sy in range(y0, y1 + 1):
                for sx in range(x0, x1 + 1):
                    idx = sy * rs + sx * 3
                    col: RGB = (raw[idx], raw[idx + 1], raw[idx + 2])
                    l = rgb_lum(col)
                    samples.append((col, l))
                    if l < min_l:
                        min_l = l
                        darkest_rgb = col
                    if l > max_l:
                        max_l = l
                        brightest_rgb = col

            contrast = max_l - min_l
            if contrast > 36.0:
                if center_l - min_l < (max_l - center_l) * 0.85:
                    grid[gy][gx] = darkest_rgb
                elif max_l - center_l < (center_l - min_l) * 0.75:
                    grid[gy][gx] = brightest_rgb
                else:
                    samples.sort(key=lambda item: item[1])
                    grid[gy][gx] = samples[len(samples) // 2][0]
            else:
                samples.sort(key=lambda item: item[1])
                grid[gy][gx] = samples[len(samples) // 2][0]

    return grid


# ==============================================================================
# MASTER 64-COLOR VGA PALETTE EXTRACTION & QUANTIZATION
# ==============================================================================
# Dedicated 14-Color Monotonic Twilight Sky Ramp (de-dithered 11x11 source averages from banner.jpg)
SKY_PALETTE_RAMP: List[RGB] = [
    (42, 48, 66),     # 0: Left zenith night slate-indigo
    (56, 56, 90),     # 1: Upper zenith twilight indigo
    (63, 60, 102),    # 2: Deep twilight blue-violet
    (79, 70, 106),    # 3: Dusk slate-violet
    (99, 74, 105),    # 4: Muted plum-lavender
    (125, 79, 106),   # 5: Dusty mauve-rose
    (148, 86, 111),   # 6: Rose-violet
    (174, 94, 107),   # 7: Warm rose
    (204, 112, 108),  # 8: Coral salmon
    (226, 129, 102),  # 9: Warm peach-coral
    (236, 153, 102),  # 10: Golden apricot
    (242, 180, 114),  # 11: Horizon amber-gold
    (250, 206, 134),  # 12: Bright solar gold
    (254, 228, 162),  # 13: Solar horizon cream
]


def extract_master_palette(grid: List[List[RGB]], n_colors: int = 64) -> List[RGB]:
    rng = random.Random(1989)
    samples: List[RGB] = []

    for y in range(NATIVE_H):
        for x in range(NATIVE_W):
            col = grid[y][x]
            if (x + y) % 3 == 0:
                samples.append(col)
            if (115 <= x <= 190 and 15 <= y <= 160) or (20 <= x <= 140 and 70 <= y <= 165):
                samples.append(col)
                if rgb_lum(col) > 175 or rgb_lum(col) < 30:
                    samples.append(col)
            if 300 <= x <= 400 and 155 <= y <= 245:
                samples.append(col)

    # Seed the 64-color VGA palette with the 14 dedicated sky ramp colors + polished brass/mahogany + cairn stone ramp
    theo_rgb_list: List[RGB] = [(v[0], v[1], v[2]) for v in THEO_PAL.values()]
    cairn_rgb_list: List[RGB] = [(v[0], v[1], v[2]) for v in CAIRN_PAL.values()]
    centroids: List[RGB] = list(SKY_PALETTE_RAMP) + theo_rgb_list
    for col in cairn_rgb_list:
        if len(centroids) < n_colors - 8 and all(rgb_dist2(col, c) > 220.0 for c in centroids):
            centroids.append(col)
    fixed_count = len(centroids)
    while len(centroids) < n_colors:
        best_c = samples[0]
        best_d = -1.0
        for _ in range(180):
            cand = samples[rng.randrange(len(samples))]
            d_min = min(rgb_dist2(cand, c) for c in centroids)
            if d_min > best_d:
                best_d = d_min
                best_c = cand
        centroids.append(best_c)

    sub = [samples[i] for i in range(0, len(samples), 2)]
    for _ in range(12):
        sums = [[0.0, 0.0, 0.0, 0] for _ in range(n_colors)]
        for col in sub:
            best_idx = 0
            best_d = 1e18
            for idx, c in enumerate(centroids):
                d = rgb_dist2(col, c)
                if d < best_d:
                    best_d = d
                    best_idx = idx
            acc = sums[best_idx]
            acc[0] += col[0]
            acc[1] += col[1]
            acc[2] += col[2]
            acc[3] += 1
        # Keep the 14 sky ramp colors and 2 anchor extremes locked; refine the 48 terrain/prop centroids
        for idx in range(fixed_count, n_colors):
            cnt = sums[idx][3]
            if cnt > 0:
                centroids[idx] = (
                    int(round(sums[idx][0] / cnt)),
                    int(round(sums[idx][1] / cnt)),
                    int(round(sums[idx][2] / cnt)),
                )

    centroids.sort(key=lambda c: (int(rgb_lum(c) // 32), c[0] - c[2], rgb_lum(c)))
    return centroids


def quantize_pixel_bayer(x: int, y: int, col: RGB, palette: Sequence[RGB], dither_strength: float = 0.35) -> RGBA:
    best1_idx = 0
    best2_idx = 0
    d1 = 1e18
    d2 = 1e18
    for i, p in enumerate(palette):
        d = rgb_dist2(col, p)
        if d < d1:
            d2 = d1
            best2_idx = best1_idx
            d1 = d
            best1_idx = i
        elif d < d2:
            d2 = d
            best2_idx = i

    p1 = palette[best1_idx]
    p2 = palette[best2_idx]

    lum = rgb_lum(col)
    if dither_strength <= 0.0 or lum < 34.0 or lum > 225.0 or abs(rgb_lum(p1) - rgb_lum(p2)) > 42.0:
        return (p1[0], p1[1], p1[2], 255)

    dist1 = math.sqrt(max(0.0, d1))
    dist2 = math.sqrt(max(1e-6, d2))
    ratio = dist1 / (dist1 + dist2)
    prob_p2 = ratio * dither_strength * 1.6
    threshold = (BAYER_4X4[y & 3][x & 3] + 0.5) / 16.0
    chosen = p2 if prob_p2 > threshold else p1
    return (chosen[0], chosen[1], chosen[2], 255)


# ==============================================================================
# GEOMETRY & EXACT BOUNDARY CONTOURS
# ==============================================================================
def point_in_poly(x: float, y: float, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def dist_point_to_seg(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    vx, vy = x1 - x0, y1 - y0
    wx, wy = px - x0, py - y0
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x0, py - y0)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x1, py - y1)
    b = c1 / c2
    bx, by = x0 + b * vx, y0 + b * vy
    return math.hypot(px - bx, py - by)


def interp_y(pts: Sequence[Point], x: float) -> float:
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if x1 <= x <= x2:
            if abs(x2 - x1) < 1e-6:
                return y1
            return y1 + (x - x1) / (x2 - x1) * (y2 - y1)
    return pts[-1][1]


def interp_x(pts: Sequence[Point], y: float) -> float:
    if y <= pts[0][1]:
        return pts[0][0]
    if y >= pts[-1][1]:
        return pts[-1][0]
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if y1 <= y <= y2:
            if abs(y2 - y1) < 1e-6:
                return x1
            return x1 + (y - y1) / (y2 - y1) * (x2 - x1)
    return pts[-1][0]


# Pixel-verified Mountain Skyline Contour (x=185..479, zero jitter)
MTN_SKYLINE_POLY: List[Point] = [
    (185, 123),
    (195, 119),
    (200, 117),
    (206, 113),
    (210, 114),
    (215, 110),
    (220, 112),
    (225, 113),
    (230, 110),
    (235, 111),
    (240, 107),
    (245, 105),
    (250, 100),
    (255, 95),
    (260, 89),
    (265, 80),
    (270, 73),
    (275, 67),
    (280, 64),
    (284, 55),
    (287, 55),
    (291, 60),
    (295, 66),
    (299, 73),
    (304, 79),
    (312, 85),
    (321, 93),
    (328, 98),
    (335, 91),
    (342, 89),
    (350, 86),
    (358, 83),
    (368, 75),
    (376, 82),
    (386, 79),
    (395, 83),
    (406, 78),
    (416, 83),
    (427, 93),
    (431, 93),
    (438, 89),
    (448, 96),
    (455, 93),
    (465, 98),
    (471, 97),
    (479, 102),
]


def get_mountain_skyline_y(grid: List[List[RGB]], x: int) -> int:
    return int(round(interp_y(MTN_SKYLINE_POLY, x)))


# Re-drawn Mid-Ground Ridge Contour capturing all crags, sunlit outcrops, and right ridge slope
MID_RIDGE_POLY: List[Point] = [
    (236, 189),
    (242, 181),
    (249, 171),
    (257, 165),
    (264, 160),
    (268, 160),
    (275, 165),
    (284, 168),
    (294, 170),
    (301, 168),
    (308, 161),
    (317, 154),
    (325, 150),
    (329, 150),
    (334, 155),
    (337, 163),
    (341, 166),
    (346, 164),
    (352, 160),
    (358, 161),
    (365, 165),
    (373, 173),
    (379, 177),
    (385, 178),
    (393, 182),
    (398, 190),
    (402, 193),
    (411, 196),
    (417, 203),
    (421, 214),
    (425, 222),
    (434, 229),
    (445, 234),
    (455, 241),
    (466, 251),
    (478, 268),
    (479, 269),
]


def get_mid_ridge_y(grid: List[List[RGB]], x: int) -> int:
    return int(round(interp_y(MID_RIDGE_POLY, x)))


CLIFF_TOP_LEFT_POLY: List[Point] = [
    (0, 102),
    (16, 103),
    (26, 109),
    (42, 116),
    (118, 118),
    (145, 119),
    (166, 119),
    (186, 122),
    (206, 124),
    (219, 129),
    (231, 132),
    (236, 138),
]

CLIFF_RIGHT_WALL_POLY: List[Point] = [
    (236, 138),
    (238, 152),
    (236, 168),
    (241, 184),
    (244, 202),
    (241, 218),
    (244, 235),
    (239, 252),
    (241, 269),
]


# ==============================================================================
# SKY COLOR FIELD RECONSTRUCTION (Seamless across Cairn, Theodolite & Mountains)
# ==============================================================================
def build_sky_color_field(
    grid: List[List[RGB]],
    w: int,
    h: int,
    rs: int,
    raw: bytes,
) -> List[List[RGB]]:
    """
    Reconstructs a completely seamless, ghost-free, moiré-free 2D twilight sky color field
    across y=0..154 by sampling three unoccluded vertical reference columns from `raw`
    (`x_left = 10`, `x_mid = 212`, `x_right = 468`) using an 11x11 source box filter
    and smoothly blending them across `x in [0..479]`.
    This guarantees zero ghost imprint of the Matterhorn, Cairn, or Theodolite on the sky layer!
    """
    def sample_source_box(gx: int, gy: int) -> RGB:
        yc = int(round(CROP_Y0 + (gy + 0.5) * CROP_H / float(NATIVE_H)))
        xc = int(round(CROP_X0 + (gx + 0.5) * CROP_W / float(NATIVE_W)))
        r_acc = g_acc = b_acc = cnt = 0
        for sy in range(max(CROP_Y0, yc - 5), min(CROP_Y0 + CROP_H, yc + 6)):
            row_off = sy * rs
            for sx in range(max(CROP_X0, xc - 5), min(CROP_X0 + CROP_W, xc + 6)):
                idx = row_off + sx * 3
                r_acc += raw[idx]
                g_acc += raw[idx + 1]
                b_acc += raw[idx + 2]
                cnt += 1
        return (r_acc // cnt, g_acc // cnt, b_acc // cnt)

    # Sample the 3 clean unoccluded vertical sky profiles (clamping y before terrain horizon)
    # Left profile at x=10 (unoccluded down to y=95)
    # Mid profile at x=212 (unoccluded down to y=104)
    # Right profile at x=468 (unoccluded down to y=90)
    prof_left: List[RGB] = []
    prof_mid: List[RGB] = []
    prof_right: List[RGB] = []
    for y in range(155):
        prof_left.append(sample_source_box(10, min(y, 94)))
        prof_mid.append(sample_source_box(212, min(y, 104)))
        prof_right.append(sample_source_box(468, min(y, 90)))

    # Smooth each vertical profile along y (radius 6) to eliminate any residual step
    def smooth_1d(prof: List[RGB]) -> List[RGB]:
        out: List[RGB] = []
        for y in range(len(prof)):
            r_s = g_s = b_s = c_s = 0
            for oy in range(-6, 7):
                ny = max(0, min(len(prof) - 1, y + oy))
                r_s += prof[ny][0]
                g_s += prof[ny][1]
                b_s += prof[ny][2]
                c_s += 1
            out.append((r_s // c_s, g_s // c_s, b_s // c_s))
        return out

    prof_left = smooth_1d(prof_left)
    prof_mid = smooth_1d(prof_mid)
    prof_right = smooth_1d(prof_right)

    smooth_sky: List[List[RGB]] = [[(0, 0, 0) for _ in range(NATIVE_W)] for _ in range(155)]
    for y in range(155):
        cL = prof_left[y]
        cM = prof_mid[y]
        cR = prof_right[y]
        for x in range(NATIVE_W):
            if x <= 212:
                t = (x / 212.0) ** 1.15
                c0, c1 = cL, cM
            else:
                t = ((x - 212.0) / float(NATIVE_W - 1 - 212)) ** 1.15
                c0, c1 = cM, cR
            smooth_sky[y][x] = (
                int(round(c0[0] + t * (c1[0] - c0[0]))),
                int(round(c0[1] + t * (c1[1] - c0[1]))),
                int(round(c0[2] + t * (c1[2] - c0[2]))),
            )

    return smooth_sky


# ==============================================================================
# SOLID, SPECKLE-FREE FOREGROUND PROP SEGMENTATION MASKS
# ==============================================================================
def is_platform_slate_or_shadow(col: RGB) -> bool:
    r, g, b = col
    lum = rgb_lum(col)
    if b >= r - 5 and 48.0 <= lum <= 178.0:
        return True
    return False


def fill_scanline_spans(
    mask: List[List[bool]],
    y_min: int,
    y_max: int,
    x_min_bound: int,
    x_max_bound: int,
    exclude_masks: Sequence[List[List[bool]]] = (),
) -> None:
    """
    On each scanline y in [y_min..y_max], fills the interior between the leftmost
    and rightmost True pixels in [x_min_bound..x_max_bound] so the object silhouette
    has zero internal pinhole gaps.
    """
    for y in range(y_min, y_max + 1):
        xs = [x for x in range(x_min_bound, x_max_bound + 1) if mask[y][x]]
        if len(xs) >= 2:
            x0, x1 = min(xs), max(xs)
            for x in range(x0, x1 + 1):
                if not any(ex[y][x] for ex in exclude_masks):
                    mask[y][x] = True


def despike_mask(mask: List[List[bool]], min_neighbors: int = 2) -> None:
    """Removes isolated 1px speckles that have fewer than `min_neighbors` 8-connected neighbors."""
    mh = len(mask)
    mw = len(mask[0]) if mh else 0
    to_clear: List[Tuple[int, int]] = []
    for y in range(1, mh - 1):
        for x in range(1, mw - 1):
            if mask[y][x]:
                nb = 0
                for oy in (-1, 0, 1):
                    for ox in (-1, 0, 1):
                        if (ox != 0 or oy != 0) and mask[y + oy][x + ox]:
                            nb += 1
                if nb < min_neighbors:
                    to_clear.append((x, y))
    for x, y in to_clear:
        mask[y][x] = False


def build_prop_masks(
    grid: List[List[RGB]],
    sky_field: List[List[RGB]],
) -> Dict[str, List[List[bool]]]:
    masks: Dict[str, List[List[bool]]] = {
        name: [[False for _ in range(NATIVE_W)] for _ in range(NATIVE_H)]
        for name in ("alpine_hut", "ice_axe", "theodolite", "notebook", "gloves", "rope", "cairn")
    }

    # 1. ALPINE HUT (x=340..351, y=196..206)
    hut_poly: List[Point] = [
        (340, 200), (344, 196), (351, 197), (351, 206), (340, 206),
    ]
    for y in range(196, 207):
        for x in range(340, 352):
            if point_in_poly(x, y, hut_poly):
                col = grid[y][x]
                if not (col[1] > col[0] + 6 and col[1] > col[2] + 8):
                    masks["alpine_hut"][y][x] = True
    fill_scanline_spans(masks["alpine_hut"], 197, 205, 340, 351)

    # 2. ICE AXE (head at y=69..82, shaft+spike to (47, 155))
    axe_head_poly: List[Point] = [
        (56, 69), (68, 69), (75, 71), (84, 74), (92, 80),
        (90, 82), (82, 79), (75, 76), (68, 74), (56, 73),
    ]
    for y in range(68, 157):
        for x in range(45, 93):
            in_head = point_in_poly(x, y, axe_head_poly)
            d_shaft = dist_point_to_seg(x, y, 71.5, 73.0, 48.0, 155.0)
            in_shaft = (y >= 72) and (d_shaft <= (2.7 if y < 140 else 2.1))
            if in_head or in_shaft:
                masks["ice_axe"][y][x] = True

    # 3. THEODOLITE & TRIPOD (x=118..181, y=16..156)
    # Tight bounding polygon around the Theodolite upper head (y=16..59) to prevent any stray sky pixels
    theo_head_poly: List[Point] = [
        (134, 21), (139, 16), (162, 16), (162, 24), (158, 32),
        (161, 44), (159, 59), (138, 59), (136, 44), (138, 32),
    ]
    for y in range(16, 157):
        for x in range(118, 182):
            col = grid[y][x]
            r, g, b = col
            lum = rgb_lum(col)

            # A. Upper Telescope, Yoke, Vertical Circle & Tribrach Head (y=16..59)
            if 16 <= y <= 59 and point_in_poly(x, y, theo_head_poly):
                # In the sky (y=16..59), sky pixels have blue > red + 5 and 35 <= lum <= 85.
                # Theodolite pixels are dark ink (lum < 35), warm brass/wood (r >= b - 3), or bright glint (lum > 88)
                is_sky_hole = (b > r + 5) and (35.0 <= lum <= 85.0)
                if not is_sky_hole:
                    masks["theodolite"][y][x] = True
                continue

            # B. Three Tripod Legs & Specular Glints (y=58..156)
            d_left = dist_point_to_seg(x, y, 143.0, 58.0, 120.0, 141.0)
            d_center = dist_point_to_seg(x, y, 147.8, 58.0, 144.8, 155.0)
            d_right = dist_point_to_seg(x, y, 152.5, 58.0, 178.0, 139.0)
            min_leg_d = min(d_left, d_center, d_right)

            in_glint = (
                math.hypot(x - 160.5, y - 77.0) <= 3.0
                or math.hypot(x - 147.5, y - 99.0) <= 3.8
                or math.hypot(x - 166.5, y - 98.0) <= 3.0
            )
            max_w = 3.4 if y < 136 else 2.4
            if min_leg_d <= max_w or in_glint:
                if y < 118:
                    is_sky = (b > r + 6) and (36.0 <= lum <= 115.0) and (min_leg_d > 1.8)
                    if not is_sky:
                        masks["theodolite"][y][x] = True
                else:
                    if not is_platform_slate_or_shadow(col) or min_leg_d <= 2.0:
                        masks["theodolite"][y][x] = True

    despike_mask(masks["theodolite"], min_neighbors=2)

    # 4. NOTEBOOK & PEN (x=151..189, y=135..155)
    notebook_poly: List[Point] = [
        (152, 144), (171, 136), (188, 143), (188, 147), (169, 155), (152, 148),
    ]
    for y in range(135, 156):
        for x in range(151, 189):
            if point_in_poly(x, y, notebook_poly) and not masks["theodolite"][y][x]:
                col = grid[y][x]
                if not is_platform_slate_or_shadow(col):
                    masks["notebook"][y][x] = True
    fill_scanline_spans(masks["notebook"], 137, 153, 152, 188, exclude_masks=[masks["theodolite"]])
    despike_mask(masks["notebook"], min_neighbors=2)

    # 5. LEATHER GLOVES (x=91..136, y=139..162)
    gloves_poly: List[Point] = [
        (91, 146), (95, 141), (106, 139), (120, 139), (133, 141),
        (136, 146), (135, 151), (126, 155), (120, 162), (110, 162),
        (103, 156), (93, 153),
    ]
    rope_tail_poly: List[Point] = [
        (79, 146), (89, 147), (95, 154), (105, 156), (105, 164), (96, 164), (90, 157), (79, 153),
    ]
    for y in range(139, 163):
        for x in range(91, 137):
            if (
                point_in_poly(x, y, gloves_poly)
                and not point_in_poly(x, y, rope_tail_poly)
                and not masks["theodolite"][y][x]
            ):
                col = grid[y][x]
                if not is_platform_slate_or_shadow(col):
                    masks["gloves"][y][x] = True
    # Fill palm & cuff interior (x=92..122) so zero pinhole gaps appear inside the gloves
    fill_scanline_spans(
        masks["gloves"], 140, 158, 92, 122,
        exclude_masks=[masks["theodolite"]],
    )
    despike_mask(masks["gloves"], min_neighbors=2)

    # 6. COILED CLIMBING ROPE & TAIL (x=63..135, y=112..164)
    rope_coil_poly: List[Point] = [
        (63, 134), (66, 125), (78, 122), (94, 119), (107, 113),
        (122, 113), (131, 120), (135, 131), (133, 139), (118, 141),
        (95, 143), (89, 151), (72, 152), (64, 144),
    ]
    for y in range(112, 165):
        for x in range(63, 136):
            if masks["gloves"][y][x] or masks["ice_axe"][y][x]:
                continue
            col = grid[y][x]
            lum = rgb_lum(col)
            if point_in_poly(x, y, rope_coil_poly):
                if not is_platform_slate_or_shadow(col):
                    masks["rope"][y][x] = True
            elif point_in_poly(x, y, rope_tail_poly):
                # Loose rope tail: keep dark outline or warm tan rope strand
                if lum < 40.0 or col[0] > col[2] + 6:
                    masks["rope"][y][x] = True

    # Solidify the main coiled rope bundle (y=114..149) so no internal transparent holes exist
    fill_scanline_spans(
        masks["rope"], 114, 149, 63, 135,
        exclude_masks=[masks["gloves"], masks["ice_axe"]],
    )
    despike_mask(masks["rope"], min_neighbors=2)

    # 7. STACKED STONE CAIRN INCLUDING EXTENDED FOOT BEHIND THEODOLITE (x=19..139, y=16..144)
    cairn_lower_poly: List[Point] = [
        (36, 98), (102, 98), (114, 102), (138, 112), (139, 124),
        (125, 130), (96, 135), (65, 141), (38, 142), (20, 139),
        (20, 126), (27, 112),
    ]
    for y in range(16, 145):
        for x in range(19, 140):
            if masks["rope"][y][x] or masks["gloves"][y][x] or masks["theodolite"][y][x]:
                continue
            col = grid[y][x]
            r, g, b = col
            lum = rgb_lum(col)
            if y <= 103 and 34 <= x <= 108:
                if lum < 34.0 or r > b + 8:
                    masks["cairn"][y][x] = True
            elif y > 98 and point_in_poly(x, y, cairn_lower_poly):
                if not is_platform_slate_or_shadow(col) or (34 <= x <= 102 and y <= 134):
                    masks["cairn"][y][x] = True

    fill_scanline_spans(
        masks["cairn"], 16, 142, 19, 139,
        exclude_masks=[masks["rope"], masks["gloves"]],
    )
    despike_mask(masks["cairn"], min_neighbors=2)

    return masks


# ==============================================================================
# LAYER & SPRITE ASSET RENDERERS (Quantized to Master 64-Color VGA Palette)
# ==============================================================================
def draw_crescent_moon(c: Canvas, cx: float = 222.0, cy: float = 34.0) -> None:
    """
    Renders a crisp VGA pixel-art crescent moon in the upper-center twilight sky,
    with its sunlit limb oriented down-right toward the alpenglow horizon and subtle
    earthshine on the shadowed portion of the lunar disc.
    """
    r_outer = 8.2
    # Shadow disc center shifted up-left so crescent faces down-right toward the sun
    sx = cx - 3.5
    sy = cy - 1.8
    r_inner = 7.3

    moon_bright: RGBA = (254, 244, 204, 255)
    moon_gold: RGBA = (247, 214, 154, 255)
    moon_warm: RGBA = (214, 172, 132, 255)
    earthshine_rim: RGBA = (74, 70, 112, 255)
    earthshine_fill: RGBA = (62, 60, 98, 255)

    x0 = int(math.floor(cx - r_outer - 1))
    x1 = int(math.ceil(cx + r_outer + 1))
    y0 = int(math.floor(cy - r_outer - 1))
    y1 = int(math.ceil(cy + r_outer + 1))

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            d_out = math.hypot(x - cx, y - cy)
            if d_out > r_outer:
                continue
            d_in = math.hypot(x - sx, y - sy)
            if d_in >= r_inner:
                # Inside the illuminated crescent!
                # Limb distance: 1.0 at outermost sunlit edge, 0.0 at inner terminator
                limb_t = min(1.0, max(0.0, (d_in - r_inner) / 3.2))
                if d_out >= r_outer - 1.05 or limb_t > 0.58:
                    c.set_px(x, y, moon_bright)
                elif limb_t > 0.22:
                    # Subtle Bayer-dithered crater shading along the mid-crescent
                    c.set_px(x, y, moon_bright if (x + y) % 2 == 0 else moon_gold)
                else:
                    c.set_px(x, y, moon_warm)
            else:
                # Inside the dark lunar disc: subtle earthshine on the upper-left rim
                if d_out >= r_outer - 0.95 and (x + y) % 2 == 0:
                    c.set_px(x, y, earthshine_rim)
                elif (x + y * 3) % 7 == 0:
                    c.set_px(x, y, earthshine_fill)


def render_sky_layer_from_field(
    grid: List[List[RGB]],
    sky_field: List[List[RGB]],
    palette: Sequence[RGB],
) -> Canvas:
    c = Canvas(NATIVE_W, NATIVE_H)
    n_stops = len(SKY_PALETTE_RAMP)

    for x in range(NATIVE_W):
        prev_t = 0.0
        for y in range(155):
            r, g, b = sky_field[y][x]
            warmth = (r - 44.0) / 202.0
            lum_prog = (rgb_lum((r, g, b)) - 52.0) / 142.0
            raw_t = 0.65 * warmth + 0.35 * lum_prog
            vert_t = ((max(0.0, y - 8.0) / 96.0) ** 1.35) * (0.62 + 0.38 * (x / float(NATIVE_W - 1)))
            t = max(prev_t, min(1.0, max(0.0, 0.72 * raw_t + 0.28 * vert_t)))
            prev_t = t

            scaled = t * (n_stops - 1)
            idx = min(n_stops - 2, int(math.floor(scaled)))
            frac = scaled - idx
            threshold = (BAYER_4X4[y & 3][x & 3] + 0.5) / 16.0
            chosen = SKY_PALETTE_RAMP[idx + 1] if frac > threshold else SKY_PALETTE_RAMP[idx]
            c.set_px(x, y, (chosen[0], chosen[1], chosen[2], 255))

    # Restore genuine isolated star pixels in the upper twilight sky (y < 44), skipping the moon area
    for y in range(4, 44):
        for x in range(6, NATIVE_W - 6):
            if (48 <= x <= 88 and y >= 15) or (133 <= x <= 163 and y >= 15) or (208 <= x <= 236 and 20 <= y <= 48):
                continue
            col = grid[y][x]
            sky_c = sky_field[y][x]
            if rgb_lum(col) - rgb_lum(sky_c) > 20.0:
                c.set_px(x, y, quantize_pixel_bayer(x, y, col, palette, dither_strength=0.0))

    # Add a few dimmer pinprick stars in the darker indigo upper-left sky (fading out toward eastern dawn)
    faint_west_stars: Sequence[Tuple[int, int, RGBA]] = (
        (22, 16, (148, 146, 178, 255)),
        (37, 46, (134, 132, 162, 255)),
        (18, 62, (122, 120, 148, 255)),
        (104, 11, (168, 166, 196, 255)),
        (116, 36, (138, 136, 166, 255)),
        (184, 19, (154, 150, 180, 255)),
    )
    for sx, sy, scol in faint_west_stars:
        c.set_px(sx, sy, scol)

    # Render the crescent moon in the upper-center twilight sky
    draw_crescent_moon(c, cx=222.0, cy=34.0)

    return c



def draw_uniform_top_border(
    c: Canvas,
    x_start: int,
    x_end: int,
    y_fn,
    border_col: RGBA = (14, 16, 28, 255),
) -> None:
    """
    Draws a continuous, uniform, 1px-thin dark pixel-art border along `(x, y_fn(x))`
    from `x_start` to `x_end`, connecting vertical steps cleanly when `|y[x] - y[x-1]| > 1`.
    """
    prev_y = y_fn(x_start)
    for x in range(x_start, x_end + 1):
        curr_y = y_fn(x)
        if abs(curr_y - prev_y) <= 1:
            c.set_px(x, curr_y, border_col)
        else:
            # Step vertically by 1px along the steeper side so the 1px border has zero gaps
            y_lo, y_hi = min(prev_y, curr_y), max(prev_y, curr_y)
            mid_y = (y_lo + y_hi) // 2
            for sy in range(y_lo, y_hi + 1):
                bx = (x - 1) if ((curr_y > prev_y and sy <= mid_y) or (curr_y < prev_y and sy > mid_y)) else x
                c.set_px(bx, sy, border_col)
        prev_y = curr_y


def render_background_mountains_layer(
    grid: List[List[RGB]],
    palette: Sequence[RGB],
) -> Canvas:
    c = Canvas(NATIVE_W, NATIVE_H)

    # Curated twilight valley-mist tones so Bayer dithering forms smooth atmospheric haze
    mist_tones: List[RGB] = [
        (46, 52, 76),
        (64, 69, 96),
        (84, 86, 114),
        (106, 102, 130),
        (126, 118, 144),
    ]
    mtn_palette: List[RGB] = list(palette) + mist_tones

    for x in range(185, NATIVE_W):
        y_top = get_mountain_skyline_y(grid, x)
        # Stop cleanly at the foreground cliff on the left (x < 238) and at the midground ridge for x >= 238
        if x < 238:
            y_bot = int(round(interp_y(CLIFF_TOP_LEFT_POLY, x))) + 2
        else:
            y_bot = get_mid_ridge_y(grid, x) + 1

        for y in range(y_top, min(NATIVE_H, y_bot)):
            src_x = x
            cliff_rx = int(round(interp_x(CLIFF_RIGHT_WALL_POLY, y))) if y >= 138 else int(round(interp_x(CLIFF_TOP_LEFT_POLY, y)))
            if x <= cliff_rx + 2:
                src_x = cliff_rx + 4 + (cliff_rx + 2 - x) % 4
            src_y = y
            if x >= 238 and y >= y_bot - 3:
                # Sample clean dark valley shadow just above the mid-ground ridge boundary
                src_y = max(y_top + 2, y_bot - 5)
            elif y <= y_top + 2:
                cand = grid[y][src_x]
                if rgb_lum(cand) < 36.0 and y + 2 < NATIVE_H:
                    src_y = min(NATIVE_H - 1, y + 2)
                elif x >= 370 and cand[0] > 215 and cand[1] > 145 and y + 2 < NATIVE_H:
                    src_y = min(NATIVE_H - 1, y + 2)
            col = grid[src_y][src_x]

            # Morning alpine valley mist pooled behind the midground ridge & across the eastern gorge
            if x >= 242 and y >= 134:
                dy_floor = max(0.0, float(y_bot - y))
                east_t = max(0.0, min(1.0, (x - 335.0) / 105.0))
                # Mist clings along the ridge saddle AND pools horizontally across the lower-right basin (y >= 170)
                mist_band_h = 34.0 + 38.0 * east_t
                floor_falloff = (1.0 - dy_floor / mist_band_h) ** 1.25 if dy_floor < mist_band_h else 0.0
                basin_falloff = max(0.0, min(1.0, (y - 166.0) / 56.0)) * east_t * 0.82
                base_falloff = max(floor_falloff, basin_falloff)
                if base_falloff > 0.02:
                    # Stratified horizontal mist ribbons via anisotropic noise
                    wisp = 0.64 + 0.36 * value_noise(x / 36.0, y / 7.5, 419)
                    # Keep bright sunlit alpine snow faces clear; veil settles in shaded valleys
                    lum_m = rgb_lum(col)
                    shade_w = max(0.20, min(1.0, (132.0 - lum_m) / 72.0))
                    max_alpha = 0.34 + 0.34 * east_t
                    mist_a = max(0.0, min(max_alpha, base_falloff * wisp * shade_w * max_alpha * 1.18))
                    if mist_a > 0.03:
                        # Cool lavender-indigo twilight mist warming subtly toward dusty lilac in the east
                        mr = 92.0 + 34.0 * east_t
                        mg = 94.0 + 22.0 * east_t
                        mb = 124.0 + 20.0 * east_t
                        col = (
                            int(round(col[0] * (1.0 - mist_a) + mr * mist_a)),
                            int(round(col[1] * (1.0 - mist_a) + mg * mist_a)),
                            int(round(col[2] * (1.0 - mist_a) + mb * mist_a)),
                        )

            c.set_px(x, y, quantize_pixel_bayer(x, y, col, mtn_palette, dither_strength=0.38))

    # Draw continuous, uniform 1px dark border along the entire mountain skyline
    draw_uniform_top_border(
        c, 185, NATIVE_W - 1,
        lambda x: get_mountain_skyline_y(grid, x),
        border_col=(14, 16, 28, 255),
    )
    return c



def render_midground_ridge_layer(
    grid: List[List[RGB]],
    masks: Dict[str, List[List[bool]]],
    palette: Sequence[RGB],
) -> Canvas:
    c = Canvas(NATIVE_W, NATIVE_H)
    hut_mask = masks["alpine_hut"]

    # Curated twilight alpine pine, moss, and sunlit sage-olive greens for ridge meadows
    alpine_greens: List[RGB] = [
        (22, 34, 34),    # deep pine-slate shadow
        (32, 46, 38),    # shaded conifer / dusky moss
        (44, 60, 44),    # weathered alpine moss
        (60, 74, 48),    # muted sage-olive turf
        (78, 90, 56),    # sunlit alpine meadow
        (104, 110, 66),  # warm dawn olive slope
        (134, 132, 82),  # golden-sage sunlit crest
    ]
    ridge_palette: List[RGB] = list(palette) + alpine_greens

    def _alpine_green_for_lum(lum: float) -> RGB:
        # Piecewise luminance-matched alpine green target so slopes keep natural shading
        stops = (
            (28.0, (22.0, 34.0, 34.0)),
            (54.0, (42.0, 58.0, 43.0)),
            (84.0, (68.0, 82.0, 52.0)),
            (122.0, (104.0, 110.0, 66.0)),
            (172.0, (144.0, 140.0, 88.0)),
        )
        if lum <= stops[0][0]:
            t = stops[0][1]
            return (int(t[0]), int(t[1]), int(t[2]))
        for i in range(len(stops) - 1):
            l0, c0 = stops[i]
            l1, c1 = stops[i + 1]
            if lum <= l1:
                u = (lum - l0) / (l1 - l0)
                return (
                    int(round(c0[0] + (c1[0] - c0[0]) * u)),
                    int(round(c0[1] + (c1[1] - c0[1]) * u)),
                    int(round(c0[2] + (c1[2] - c0[2]) * u)),
                )
        t = stops[-1][1]
        return (int(t[0]), int(t[1]), int(t[2]))

    # Top boundary of alpine vegetation zone (keeps jagged upper peak rock mostly bare)
    veg_top_curve: List[Point] = [
        (236, 198), (275, 194), (315, 198), (345, 196),
        (372, 178), (402, 190), (430, 214), (479, 240),
    ]

    for x in range(236, NATIVE_W):
        y_top = get_mid_ridge_y(grid, x)
        y_veg = interp_y(veg_top_curve, x)
        for y in range(y_top, NATIVE_H):
            src_x, src_y = x, y
            if x <= int(round(interp_x(CLIFF_RIGHT_WALL_POLY, y))) + 1:
                src_x = 247 + (245 - x) % 5
            elif hut_mask[y][x]:
                src_x = x - 11 if x < 346 else x + 9
                src_y = min(NATIVE_H - 1, y + 2)
            elif y <= y_top + 2:
                # Ensure uniform border thickness by sampling interior rock/meadow just below thick black outline pixels
                cand = grid[y][src_x]
                if rgb_lum(cand) < 28.0 and y + 2 < NATIVE_H and x >= 325:
                    src_y = min(NATIVE_H - 1, y + 2)
            col = grid[src_y][src_x]
            r, g, b = col
            lum = rgb_lum(col)

            # Blend organic muted alpine-green hues onto grassy slopes & mossy terraces
            if y >= y_veg - 6 and lum >= 26.0:
                elev_w = max(0.0, min(1.0, (y - (y_veg - 6.0)) / 18.0))
                # Preserve cold deep-blue rock shadow crags and bright warm gravel trail highlights
                rock_blue = max(0.0, min(1.0, (b - r - 6.0) / 22.0))
                scree_trail = max(0.0, min(1.0, (lum - 138.0) / 55.0)) if (290 <= x <= 362 and 212 <= y <= 246) else 0.0
                # Multi-scale organic moss/meadow patchiness so rocky ribs still peek through
                n_macro = value_noise(x / 17.0, y / 11.0, 911)
                n_detail = value_noise(x / 6.5, y / 5.5, 929)
                patch = max(0.0, min(1.0, (n_macro * 0.65 + n_detail * 0.35 - 0.28) / 0.56))
                # Stronger moss/turf presence on the sweeping slopes around the hut and right shoulder
                slope_boost = 1.0 if (x >= 352 and y <= 235) else (0.90 if y <= 244 else 0.72)
                green_w = elev_w * patch * slope_boost * (1.0 - 0.80 * rock_blue) * (1.0 - 0.75 * scree_trail)
                green_w = max(0.0, min(0.72, green_w * 0.82))
                if green_w > 0.05:
                    tgt = _alpine_green_for_lum(lum)
                    col = (
                        int(round(r * (1.0 - green_w) + tgt[0] * green_w)),
                        int(round(g * (1.0 - green_w) + tgt[1] * green_w)),
                        int(round(b * (1.0 - green_w) + tgt[2] * green_w)),
                    )

            c.set_px(x, y, quantize_pixel_bayer(x, y, col, ridge_palette, dither_strength=0.34))

    # Draw continuous, uniform 1px dark border along the entire mid-ground ridge top contour
    draw_uniform_top_border(
        c, 236, NATIVE_W - 1,
        lambda x: get_mid_ridge_y(grid, x),
        border_col=(12, 14, 24, 255),
    )
    return c



def render_foreground_cliff_layer(
    grid: List[List[RGB]],
    masks: Dict[str, List[List[bool]]],
    palette: Sequence[RGB],
) -> Canvas:
    """
    Renders the 480x270 foreground cliff layer conditioned directly on `banner.jpg`:
    preserves all clean exposed rock, uneven slab texture, pebbles, right-side
    sunlight, and snow patches from `grid`, while inpainting ONLY the prop and
    baked-shadow footprints using non-repeating multi-scale granite + frost synthesis
    seeded from the surrounding clean platform pixels.
    """
    c = Canvas(NATIVE_W, NATIVE_H)
    prop_names = ("cairn", "rope", "ice_axe", "gloves", "notebook", "theodolite")

    lip_front_1x: List[Point] = [
        (0, 155), (14, 162), (36, 161), (54, 171), (82, 171),
        (100, 179), (124, 181), (156, 170), (170, 161),
        (194, 154), (225, 145), (236, 138),
    ]

    # Build mask of pixels on the upper platform that belong to props or their baked ground shadows
    bad_plat = [[False for _ in range(NATIVE_W)] for _ in range(NATIVE_H)]
    for y in range(102, 178):
        max_x = int(round(interp_x(CLIFF_RIGHT_WALL_POLY, y))) if y >= 138 else 236
        for x in range(0, min(NATIVE_W, max_x + 1)):
            y_top = int(round(interp_y(CLIFF_TOP_LEFT_POLY, x)))
            y_lip = int(round(interp_y(lip_front_1x, x))) if x <= 236 else 138
            if y < y_top or y > y_lip:
                continue

            # Exclude foreground props and right-edge rim cobbles (extracted to 960x540 foreground sprite)
            in_prop = False
            for oy in range(-2, 3):
                ny = max(0, min(NATIVE_H - 1, y + oy))
                for ox in range(-2, 3):
                    nx = max(0, min(NATIVE_W - 1, x + ox))
                    if any(masks[k][ny][nx] for k in prop_names):
                        in_prop = True
                        break
                if in_prop:
                    break
            if in_prop or (186 <= x <= 225 and 119 <= y <= 137) or (191 <= x <= 212 and 138 <= y <= 140):
                bad_plat[y][x] = True
                continue

            col = grid[y][x]
            r, g, b = col
            lum = rgb_lum(col)
            # Detect baked dark cast shadows left behind around cairn, axe, rope, gloves, notebook, and tripod
            is_baked_shad = (
                (6 <= x <= 38 and 122 <= y <= 146 and (lum < 60.0 or (b > r + 10 and lum < 78.0)))
                or (38 <= x <= 156 and 118 <= y <= 164 and (lum < 64.0 or (b > r + 10 and lum < 82.0)))
                or (134 <= x <= 188 and 118 <= y <= 158 and (lum < 64.0 or (b > r + 10 and lum < 80.0)))
                or (80 <= x <= 108 and 150 <= y <= 167)  # trailing rope-tail area in original JPEG
            )
            # Exclude clean deep cliff lip cracks right at the brink (y >= y_lip - 1) and left ridge rock (x <= 25, y <= y_top + 6)
            if y >= y_lip - 1 or (x <= 25 and y <= y_top + 6):
                is_baked_shad = False
            if is_baked_shad:
                bad_plat[y][x] = True

    # Collect clean exposed platform pixels (`clean_pts`) and compute distance-weighted clean anchors
    clean_plat_cols: List[Tuple[int, int, RGB]] = []
    for y in range(106, 180):
        for x in range(0, 236):
            y_top = int(round(interp_y(CLIFF_TOP_LEFT_POLY, x)))
            y_lip = int(round(interp_y(lip_front_1x, x)))
            if y_top + 1 <= y <= y_lip - 1 and not bad_plat[y][x]:
                col = grid[y][x]
                if rgb_lum(col) >= 48.0:
                    clean_plat_cols.append((x, y, col))

    def _lerp_rgb(a: RGB, b: RGB, t: float) -> RGB:
        tt = max(0.0, min(1.0, t))
        return (
            int(round(a[0] + (b[0] - a[0]) * tt)),
            int(round(a[1] + (b[1] - a[1]) * tt)),
            int(round(a[2] + (b[2] - a[2]) * tt)),
        )

    # Smooth right-lit alpine granite + frost palette sampled directly from banner.jpg platform
    rock_shade: RGB = (64, 68, 82)
    rock_mid: RGB = (94, 92, 98)
    rock_warm: RGB = (118, 102, 95)
    rock_bright: RGB = (138, 122, 114)
    snow_cool: RGB = (116, 134, 156)
    snow_lit: RGB = (176, 184, 204)

    # Muted high-altitude lichen / crevice moss accents echoing the midground alpine greens
    cliff_lichen: List[RGB] = [
        (38, 48, 42),   # dark damp crevice moss
        (56, 68, 50),   # weathered sage-olive lichen
        (78, 86, 58),   # sunlit alpine rock lichen
    ]
    cliff_palette: List[RGB] = list(palette) + cliff_lichen

    for y in range(102, NATIVE_H):
        max_x = int(round(interp_x(CLIFF_RIGHT_WALL_POLY, y))) if y >= 138 else 236
        for x in range(0, min(NATIVE_W, max_x + 1)):
            y_top_cliff = int(round(interp_y(CLIFF_TOP_LEFT_POLY, x)))
            if y < y_top_cliff:
                continue

            if not bad_plat[y][x]:
                col = grid[y][x]
                y_lip_here = int(round(interp_y(lip_front_1x, x))) if x <= 236 else 138
                if y >= y_lip_here + 2 and x <= max_x - 3:
                    r, g, b = col
                    lum = rgb_lum(col)
                    is_snow = (b > r + 15 and lum > 105.0)
                    is_orange_rim = (r > b + 30)
                    if 40.0 <= lum <= 118.0 and not is_snow and not is_orange_rim:
                        near_crack = False
                        for dx, dy in ((0, 1), (0, 2), (0, -1), (-1, 1), (1, 1)):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < NATIVE_W and 0 <= ny < NATIVE_H and rgb_lum(grid[ny][nx]) < 34.0:
                                near_crack = True
                                break
                        if near_crack:
                            p_macro = value_noise(x, y, 16.0, 613)
                            p_fine = value_noise(x, y, 5.5, 641)
                            patch = p_macro * 0.65 + p_fine * 0.35
                            if patch > 0.58:
                                lichen_w = min(0.76, (patch - 0.58) / 0.24 * 0.76)
                                tgt = cliff_lichen[0] if lum < 58.0 else (cliff_lichen[1] if lum < 86.0 else cliff_lichen[2])
                                col = _lerp_rgb(col, tgt, lichen_w)
                c.set_px(x, y, quantize_pixel_bayer(x, y, col, cliff_palette, dither_strength=0.32))
                continue

            # Horizon outline along upper edge where sky/mountains sit behind removed cairn
            if y == y_top_cliff and x >= 26:
                c.set_px(x, y, quantize_pixel_bayer(x, y, (24, 22, 34), palette, dither_strength=0.0))
                continue

            # Nearest clean platform neighbor color (excluding snow outliers for rock base)
            best_rock: Optional[RGB] = None
            best_d2 = 1e18
            for cx, cy, ccol in clean_plat_cols:
                # Prefer nearby clean non-snow stone at similar x-exposure
                d2 = (cx - x) ** 2 + ((cy - y) * 1.4) ** 2
                if d2 < best_d2 and rgb_lum(ccol) < 135.0:
                    best_d2 = d2
                    best_rock = ccol

            sun_r = max(0.0, min(1.0, (x - 16.0) / 195.0))
            base_stone = _lerp_rgb(
                _lerp_rgb(rock_shade, rock_mid, sun_r),
                _lerp_rgb(rock_warm, rock_bright, sun_r),
                0.35 + 0.55 * sun_r,
            )
            if best_rock is not None:
                # Condition closely on banner.jpg's nearest clean stone pixels
                base_stone = _lerp_rgb(best_rock, base_stone, 0.45)

            # Natural uneven granite relief: multi-scale bumpy rock modulation + subtle hairline ridges
            n_broad = value_noise(x * 0.65, y * 1.25, 13.0, 401)
            n_bump = value_noise(x * 0.95, y * 1.50, 4.8, 709)
            n_grit = fbm_hash(x, y, 913)
            bump = (n_broad - 0.5) * 0.40 + (n_bump - 0.5) * 0.38 + (n_grit - 0.5) * 0.22

            # Occasional natural dark granite pocket / micro-crevice (non-repeating)
            ridge = abs(value_noise(x * 0.75, y * 1.4, 8.5, 557) - 0.5)
            if ridge < 0.032 and n_broad < 0.58:
                stone_col = _lerp_rgb(base_stone, (42, 44, 58), 0.55)
            elif bump > 0.0:
                stone_col = _lerp_rgb(base_stone, rock_bright if sun_r > 0.45 else (112, 114, 126), min(0.75, bump * 1.4))
            else:
                stone_col = _lerp_rgb(base_stone, (54, 58, 74), min(0.65, -bump * 1.3))

            # Scattered frosty snow patches matching banner.jpg's frost drifts (112,133,154 .. 181,186,206)
            snow_field = (
                value_noise(x * 0.62, y * 1.35, 11.5, 443) * 0.68
                + value_noise(x * 1.1, y * 1.8, 4.2, 881) * 0.32
            )
            if snow_field > 0.68:
                snow_t = min(1.0, (snow_field - 0.68) / 0.14)
                snow_col = _lerp_rgb(snow_cool, snow_lit, min(1.0, sun_r * 0.85 + 0.15 * n_grit))
                stone_col = _lerp_rgb(stone_col, snow_col, snow_t * 0.88)

            c.set_px(x, y, quantize_pixel_bayer(x, y, stone_col, palette, dither_strength=0.28))

    return c



# Dedicated Twilight-Muted Antique Brass, Mahogany & Glass Ramp for the 2x Foreground Theodolite
THEO_PAL: Dict[str, RGBA] = {
    "INK_SHADOW":     (18, 14, 24, 255),     # Deep twilight outline (shadow undersides & deep crevices)
    "OUTLINE_BRONZE": (54, 36, 26, 255),     # Selective colored outline (lit/interior brass edges)
    "OUTLINE_MAHOG":  (36, 20, 20, 255),     # Selective colored outline (wood legs)
    "BRASS_SHAD":     (82, 54, 34, 255),     # Dark patinated bronze shadow
    "BRASS_BRONZE":   (122, 84, 46, 255),    # Muted mid antique bronze
    "BRASS_GOLD":     (160, 118, 62, 255),   # Muted warm brass
    "BRASS_BRIGHT":   (194, 152, 86, 255),   # Soft sunlit brass
    "BRASS_GLEAM":    (220, 186, 118, 255),  # Subtle specular brass
    "BRASS_SPEC":     (242, 228, 192, 255),  # Soft warm-gold glint
    "MAHOG_DARK":     (52, 26, 26, 255),     # Shadowed weathered mahogany
    "MAHOG_MID":      (84, 44, 38, 255),     # Muted mid mahogany
    "MAHOG_LIGHT":    (120, 66, 52, 255),    # Subtle lit mahogany
    "MAHOG_RIM":      (154, 94, 68, 255),    # Soft dusk rim on wood
    "GLASS_DARK":     (34, 42, 64, 255),     # Telescope objective glass shadow
    "GLASS_GLINT":    (152, 174, 202, 255),  # Subtle objective glass reflection
}


def render_polished_theodolite_sprite_2x() -> Canvas:
    """
    Renders a crisp, sleek, high-detail pixel-art Brass Theodolite & Mahogany Tripod
    directly on the 960x540 (2x) foreground pixel grid, using selective colored
    outlines ("selout"), cylindrical shading ramps, open yoke/tribrach windows,
    and diamond specular glints.
    Anchor coordinates match 2x of banner.jpg's (119..180, 16..156) -> (238..360, 32..312).
    """
    c = Canvas(FG_W, FG_H)
    P = THEO_PAL

    def hspan(
        y: int,
        x0: int,
        x1: int,
        cols: Sequence[RGBA],
        left_out: Optional[RGBA] = P["OUTLINE_BRONZE"],
        right_out: Optional[RGBA] = P["OUTLINE_BRONZE"],
    ) -> None:
        if left_out is not None:
            c.set_px(x0 - 1, y, left_out)
        if right_out is not None:
            c.set_px(x1 + 1, y, right_out)
        span = max(1, x1 - x0)
        n = len(cols)
        for x in range(x0, x1 + 1):
            idx = min(n - 1, int(round(((x - x0) / float(span)) * (n - 1))))
            c.set_px(x, y, cols[idx])

    def draw_tapered_leg(
        x0: float,
        y0: int,
        x1: float,
        y1: int,
        profile_fn,
    ) -> None:
        """Draws a leg segment from y0..y1 using a row-by-row offset->color dict."""
        for y in range(y0, y1 + 1):
            t = (y - y0) / float(max(1, y1 - y0))
            cx = int(round(x0 + t * (x1 - x0)))
            row_map = profile_fn(t, y)
            for ox, col in row_map.items():
                c.set_px(cx + ox, y, col)

    # ==========================================================================
    # 1. THREE TRIPOD LEGS (y = 116..311 at 960x540 scale)
    # ==========================================================================
    # --- A. LEFT LEG: crown (285, 116) -> mid-clamp (262, 192) -> ferrule (244, 260) -> tip (240, 282)
    draw_tapered_leg(
        285.0, 116, 262.0, 192,
        lambda t, y: {
            -4: P["INK_SHADOW"],
            -3: P["BRASS_SHAD"],
            -2: P["BRASS_BRONZE"],
            -1: P["INK_SHADOW"] if (y < 186 and y > 122) else P["BRASS_SHAD"],  # Hollow split slot!
             0: P["BRASS_GOLD"],
             1: P["BRASS_BRIGHT"],
             2: P["BRASS_GLEAM"] if y % 7 in (0, 1) else P["BRASS_BRIGHT"],
             3: P["OUTLINE_BRONZE"],
        },
    )
    draw_tapered_leg(
        262.0, 194, 244.0, 260,
        lambda t, y: {
            -3: P["INK_SHADOW"],
            -2: P["MAHOG_DARK"],
            -1: P["MAHOG_MID"],
             0: P["MAHOG_MID"],
             1: P["MAHOG_LIGHT"],
             2: P["MAHOG_RIM"] if t < 0.7 else P["MAHOG_LIGHT"],
             3: P["OUTLINE_MAHOG"],
        },
    )
    # Left tapered brass ferrule & ground spike (y = 261..282)
    draw_tapered_leg(
        244.0, 261, 240.0, 282,
        lambda t, y: (
            {-2: P["INK_SHADOW"], -1: P["BRASS_SHAD"], 0: P["BRASS_GOLD"], 1: P["BRASS_GLEAM"], 2: P["OUTLINE_BRONZE"]}
            if t < 0.65
            else {-1: P["INK_SHADOW"], 0: P["BRASS_BRIGHT"], 1: P["BRASS_SPEC"], 2: P["OUTLINE_BRONZE"]}
        ),
    )
    c.set_px(240, 283, P["INK_SHADOW"])

    # Left leg brass mid-clamp collar & wing screw (y = 188..195, cx = 262)
    hspan(188, 259, 265, [P["OUTLINE_BRONZE"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(189, 258, 266, [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(190, 258, 266, [P["BRASS_BRONZE"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(191, 258, 266, [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(192, 258, 266, [P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_BRIGHT"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(193, 259, 265, [P["INK_SHADOW"], P["BRASS_SHAD"], P["BRASS_BRONZE"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    # Small clamping screw lug on left side of collar
    c.set_px(256, 190, P["BRASS_GOLD"])
    c.set_px(256, 191, P["BRASS_SHAD"])
    c.set_px(255, 190, P["INK_SHADOW"])
    c.set_px(255, 191, P["INK_SHADOW"])

    # --- B. RIGHT LEG: crown (306, 116) -> mid-clamp (332, 196) -> ferrule (351, 258) -> tip (356, 278)
    draw_tapered_leg(
        306.0, 116, 332.0, 195,
        lambda t, y: {
            -3: P["OUTLINE_BRONZE"],
            -2: P["BRASS_BRONZE"],
            -1: P["BRASS_GOLD"],
             0: P["OUTLINE_BRONZE"] if (y < 188 and y > 122) else P["BRASS_GOLD"],  # Split strut slot
             1: P["BRASS_BRIGHT"],
             2: P["BRASS_GLEAM"],
             3: P["BRASS_SPEC"] if y % 6 in (0, 1) else P["BRASS_GLEAM"],
             4: P["OUTLINE_BRONZE"],
        },
    )
    draw_tapered_leg(
        332.0, 197, 351.0, 258,
        lambda t, y: {
            -3: P["OUTLINE_MAHOG"],
            -2: P["MAHOG_DARK"],
            -1: P["MAHOG_MID"],
             0: P["MAHOG_MID"],
             1: P["MAHOG_LIGHT"],
             2: P["MAHOG_RIM"],
             3: P["OUTLINE_MAHOG"],
        },
    )
    # Right leg brass ferrule & spike (y = 259..278)
    draw_tapered_leg(
        351.0, 259, 356.0, 278,
        lambda t, y: (
            {-2: P["OUTLINE_BRONZE"], -1: P["BRASS_BRONZE"], 0: P["BRASS_BRIGHT"], 1: P["BRASS_SPEC"], 2: P["OUTLINE_BRONZE"]}
            if t < 0.65
            else {-1: P["OUTLINE_BRONZE"], 0: P["BRASS_GLEAM"], 1: P["BRASS_SPEC"], 2: P["OUTLINE_BRONZE"]}
        ),
    )
    c.set_px(356, 279, P["INK_SHADOW"])

    # Right leg upper brass tie band at y = 152..156 (cx = 318)
    for dy, cols in enumerate([
        [P["BRASS_BRONZE"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]],
        [P["BRASS_GOLD"], P["BRASS_GLEAM"], P["BRASS_SPEC"], P["BRASS_SPEC"]],
        [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"]],
    ]):
        hspan(153 + dy, 315, 321, cols, P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])

    # Right leg mid-clamp collar (y = 193..199, cx = 332)
    hspan(193, 329, 335, [P["OUTLINE_BRONZE"]])
    hspan(194, 328, 336, [P["BRASS_BRONZE"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    hspan(195, 328, 336, [P["BRASS_GOLD"], P["BRASS_GLEAM"], P["BRASS_SPEC"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    hspan(196, 328, 336, [P["BRASS_BRONZE"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    hspan(197, 328, 336, [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    hspan(198, 329, 335, [P["INK_SHADOW"], P["BRASS_SHAD"], P["BRASS_BRONZE"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    # Wing screw on right side
    c.set_px(338, 195, P["BRASS_GLEAM"])
    c.set_px(338, 196, P["BRASS_GOLD"])
    c.set_px(339, 195, P["OUTLINE_BRONZE"])
    c.set_px(339, 196, P["OUTLINE_BRONZE"])

    # --- C. CENTER FOREGROUND LEG: crown (295, 116) -> mid-clamp (293, 195) -> ferrule (290, 276) -> tip (289, 310)
    # Upper split-rail leg (y = 116..192)
    draw_tapered_leg(
        295.0, 116, 293.0, 192,
        lambda t, y: {
            -4: P["INK_SHADOW"],
            -3: P["MAHOG_DARK"],
            -2: P["BRASS_BRONZE"],
            -1: P["BRASS_GOLD"],
             0: P["INK_SHADOW"],  # Clean 1px hollow center slot
             1: P["BRASS_BRIGHT"],
             2: P["BRASS_GLEAM"],
             3: P["BRASS_SPEC"] if y % 9 == 0 else P["BRASS_GLEAM"],
             4: P["OUTLINE_BRONZE"],
        },
    )
    # Lower mahogany leg (y = 199..275), smooth continuous profile into the ferrule collar
    draw_tapered_leg(
        293.0, 199, 290.0, 275,
        lambda t, y: {
            -3: P["INK_SHADOW"],
            -2: P["MAHOG_DARK"],
            -1: P["MAHOG_MID"],
             0: P["MAHOG_LIGHT"],
             1: P["MAHOG_RIM"] if t < 0.55 else P["MAHOG_LIGHT"],
             2: P["BRASS_BRIGHT"],
             3: P["OUTLINE_MAHOG"],
        },
    )
    # Center leg brass mid-clamp collar & wing screw (y = 192..199, cx = 293)
    hspan(192, 289, 297, [P["OUTLINE_BRONZE"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(193, 288, 298, [P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(194, 288, 298, [P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_SPEC"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(195, 288, 298, [P["BRASS_BRONZE"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"], P["BRASS_GLEAM"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(196, 288, 298, [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_GOLD"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(197, 289, 297, [P["INK_SHADOW"], P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_GOLD"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    # Wing nut on right side of center clamp
    c.set_px(300, 194, P["BRASS_SPEC"])
    c.set_px(300, 195, P["BRASS_GLEAM"])
    c.set_px(301, 194, P["OUTLINE_BRONZE"])
    c.set_px(301, 195, P["OUTLINE_BRONZE"])

    # Center leg lower brass ferrule collar (y = 276..281) & tapered brass ground spike (y = 282..310)
    hspan(276, 287, 293, [P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(277, 286, 294, [P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(278, 286, 294, [P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_GLEAM"], P["BRASS_SPEC"], P["BRASS_GLEAM"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(279, 287, 293, [P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_BRIGHT"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(280, 288, 292, [P["INK_SHADOW"], P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_GOLD"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    for y in range(281, 311):
        if y <= 292:
            hspan(y, 288, 291, [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
        elif y <= 303:
            hspan(y, 288, 290, [P["BRASS_GOLD"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
        elif y <= 308:
            hspan(y, 288, 289, [P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
        else:
            hspan(y, 289, 289, [P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    c.set_px(289, 311, P["INK_SHADOW"])

    # ==========================================================================
    # 2. TRIPOD CROWN, TRIBRACH & HORIZONTAL VERNIER PLATES (y = 78..116, cx = 296)
    # ==========================================================================
    brass_cyl = [
        P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_GOLD"],
        P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"], P["BRASS_GLEAM"],
    ]
    brass_rim_dark = [
        P["OUTLINE_BRONZE"], P["BRASS_SHAD"], P["BRASS_BRONZE"],
        P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GOLD"],
    ]

    # Lower tripod crown collar (y = 108..115)
    hspan(115, 281, 311, [P["INK_SHADOW"]], P["INK_SHADOW"], P["INK_SHADOW"])
    hspan(114, 280, 312, brass_rim_dark, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    for y in range(110, 114):
        hspan(y, 280, 312, brass_cyl, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(109, 281, 311, [P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(108, 282, 310, [P["OUTLINE_BRONZE"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])

    # Recessed neck & middle tribrach flange (y = 97..107)
    for y in range(105, 108):
        hspan(y, 283, 309, brass_cyl, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(104, 281, 311, [P["INK_SHADOW"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(103, 279, 313, brass_rim_dark, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    for y in range(100, 103):
        hspan(y, 279, 313, brass_cyl, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(99, 280, 312, [P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(98, 282, 310, [P["OUTLINE_BRONZE"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])

    # Three turned brass leveling thumbscrews with crisp knurled caps & open sky gaps (y = 87..97)
    for y in range(87, 98):
        # Knurled wheel disc around mid-height (y = 90..93)
        is_knurl = 90 <= y <= 93
        lx0, lx1 = (280, 285) if is_knurl else (281, 284)
        cx0, cx1 = (293, 299) if is_knurl else (294, 298)
        rx0, rx1 = (307, 312) if is_knurl else (308, 311)
        hspan(y, lx0, lx1, [P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_BRIGHT"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
        hspan(y, cx0, cx1, [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
        hspan(y, rx0, rx1, [P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])

    # Wide upper horizontal azimuth / vernier circle plate (y = 78..86, x = 272..320)
    hspan(86, 274, 318, [P["INK_SHADOW"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(85, 273, 319, brass_rim_dark, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    for y in range(81, 85):
        hspan(y, 272, 320, brass_cyl, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
        # Subtle graduated vernier tick marks along the lower half of the bevel
        if y == 83:
            for tx in range(276, 317, 4):
                c.set_px(tx, y, P["OUTLINE_BRONZE"])
    hspan(80, 273, 319, [P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    hspan(79, 274, 318, [P["OUTLINE_BRONZE"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])

    # ==========================================================================
    # 3. SLEEK ALIDADE YOKE STANDARDS, SPIRIT LEVEL & VERTICAL CIRCLE (y = 46..78)
    # ==========================================================================
    # Compass housing base on top of the azimuth plate (y = 73..78)
    hspan(78, 281, 311, [P["BRASS_SHAD"], P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    for y in range(74, 78):
        hspan(y, 281, 311, brass_cyl, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(73, 282, 310, [P["OUTLINE_BRONZE"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])

    # Horizontal spirit-level crossbar (y = 66..69) with delicate glass vial highlight
    hspan(69, 279, 313, [P["INK_SHADOW"]], P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(68, 278, 314, brass_cyl, P["INK_SHADOW"], P["OUTLINE_BRONZE"])
    hspan(67, 278, 314, [P["BRASS_GOLD"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    hspan(66, 279, 313, [P["OUTLINE_BRONZE"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    # Glass spirit bubble accent in center of bar
    hspan(67, 293, 299, [P["GLASS_DARK"], P["GLASS_GLINT"], P["BRASS_SPEC"], P["GLASS_GLINT"]], None, None)

    # Slanting Left & Right A-Frame Standards connecting cleanly up to the telescope trunnion (y = 43..72)
    draw_tapered_leg(
        286.5, 46, 281.0, 72,
        lambda t, y: {
            -2: P["INK_SHADOW"],
            -1: P["BRASS_SHAD"],
             0: P["BRASS_GOLD"],
             1: P["BRASS_BRIGHT"],
             2: P["OUTLINE_BRONZE"],
        },
    )
    draw_tapered_leg(
        304.0, 42, 310.0, 72,
        lambda t, y: {
            -2: P["OUTLINE_BRONZE"],
            -1: P["BRASS_GOLD"],
             0: P["BRASS_BRIGHT"],
             1: P["BRASS_GLEAM"],
             2: P["BRASS_SPEC"] if y < 60 else P["BRASS_GLEAM"],
             3: P["OUTLINE_BRONZE"],
        },
    )
    # Central vertical axis spindle (x = 295..296, y = 44..72)
    for y in range(44, 73):
        if y in (66, 67, 68, 69):
            continue
        hspan(y, 295, 296, [P["BRASS_BRONZE"], P["BRASS_GLEAM"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])

    # Delicate Graduated Vertical Circle Ring (cx = 292.5, cy = 58.5, r_out = 9.8, r_in = 6.2)
    vcx, vcy = 292.5, 58.5
    for y in range(47, 70):
        for x in range(281, 305):
            d = math.hypot(x - vcx, y - vcy)
            if 6.0 <= d <= 9.0:
                nx = (x - vcx) / 9.0
                ny = (y - vcy) / 9.0
                if nx > 0.35 and ny < 0.25:
                    col = P["BRASS_SPEC"]
                elif nx > 0.0:
                    col = P["BRASS_GLEAM"]
                elif ny < 0.0:
                    col = P["BRASS_BRIGHT"]
                else:
                    col = P["BRASS_GOLD"]
                c.set_px(x, y, col)
            elif 9.0 < d <= 10.2 or 4.9 <= d < 6.0:
                c.set_px(x, y, P["OUTLINE_BRONZE"])
    # Crossbar & central trunnion bearing boss inside vertical circle
    hspan(58, 286, 299, [P["BRASS_BRONZE"], P["BRASS_GOLD"], P["BRASS_GLEAM"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    hspan(59, 286, 299, [P["BRASS_SHAD"], P["BRASS_BRONZE"], P["BRASS_BRIGHT"], P["BRASS_GLEAM"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])
    for y in range(56, 61):
        hspan(y, 291, 294, [P["BRASS_BRONZE"], P["BRASS_BRIGHT"], P["BRASS_SPEC"]], P["OUTLINE_BRONZE"], P["OUTLINE_BRONZE"])

    # ==========================================================================
    # 4. SIGHTING TELESCOPE BARREL (tilted upward-right from (268, 46) to (323, 34))
    # ==========================================================================
    for x in range(268, 324):
        u = (x - 268) / 55.0
        cy = 46.8 - u * 11.4
        iy = int(round(cy))

        # Flared objective sunshade bell on right (x >= 314)
        if x >= 314:
            half_h = 5 if x >= 318 else 4
            c.set_px(x, iy - half_h - 1, P["OUTLINE_BRONZE"])
            c.set_px(x, iy - half_h,     P["BRASS_SPEC"])
            for dy in range(-half_h + 1, 0):
                c.set_px(x, iy + dy, P["BRASS_GLEAM"] if x >= 317 else P["BRASS_BRIGHT"])
            c.set_px(x, iy,     P["BRASS_BRIGHT"])
            c.set_px(x, iy + 1, P["BRASS_GOLD"])
            for dy in range(2, half_h):
                c.set_px(x, iy + dy, P["BRASS_BRONZE"])
            c.set_px(x, iy + half_h,     P["BRASS_SHAD"])
            c.set_px(x, iy + half_h + 1, P["INK_SHADOW"])
        # Eyepiece focus tube & stepped collar on left (x <= 277)
        elif x <= 277:
            half_h = 3 if x <= 273 else 4
            c.set_px(x, iy - half_h - 1, P["OUTLINE_BRONZE"])
            c.set_px(x, iy - half_h,     P["BRASS_GLEAM"])
            for dy in range(-half_h + 1, 1):
                c.set_px(x, iy + dy, P["BRASS_BRIGHT"])
            for dy in range(1, half_h):
                c.set_px(x, iy + dy, P["BRASS_GOLD"] if dy == 1 else P["BRASS_BRONZE"])
            c.set_px(x, iy + half_h,     P["BRASS_SHAD"])
            c.set_px(x, iy + half_h + 1, P["INK_SHADOW"])
        # Main tapered brass optical barrel (x = 278..313)
        else:
            c.set_px(x, iy - 5, P["OUTLINE_BRONZE"])
            c.set_px(x, iy - 4, P["BRASS_SPEC"] if x >= 295 else P["BRASS_GLEAM"])
            c.set_px(x, iy - 3, P["BRASS_SPEC"] if x >= 304 else P["BRASS_GLEAM"])
            c.set_px(x, iy - 2, P["BRASS_GLEAM"] if x >= 294 else P["BRASS_BRIGHT"])
            c.set_px(x, iy - 1, P["BRASS_BRIGHT"])
            c.set_px(x, iy,     P["BRASS_GOLD"])
            c.set_px(x, iy + 1, P["BRASS_BRONZE"])
            c.set_px(x, iy + 2, P["BRASS_SHAD"])
            c.set_px(x, iy + 3, P["BRASS_SHAD"])
            c.set_px(x, iy + 4, P["INK_SHADOW"])

    # Small brass clamping rings around telescope barrel at x=286..288 and x=304..306
    for rx in (286, 287, 304, 305):
        u = (rx - 268) / 55.0
        iy = int(round(46.8 - u * 11.4))
        c.set_px(rx, iy - 5, P["BRASS_SPEC"])
        c.set_px(rx, iy + 4, P["INK_SHADOW"])

    # Eyepiece rim (x = 266..267) & Objective sapphire glass lens reflection (x = 324..325)
    for dy in range(-3, 4):
        c.set_px(267, 47 + dy, P["OUTLINE_BRONZE"])
    c.set_px(266, 46, P["GLASS_DARK"])
    c.set_px(266, 47, P["GLASS_GLINT"])

    for dy in range(-5, 6):
        c.set_px(324, 35 + dy, P["OUTLINE_BRONZE"] if dy < 0 else P["INK_SHADOW"])
    # Subtle glass glint inside the objective aperture on the front-right rim
    for dy in (-3, -2, -1, 0, 1, 2):
        c.set_px(323, 35 + dy, P["GLASS_GLINT"] if dy < 0 else P["GLASS_DARK"])

    # ==========================================================================
    # 5. SIGNATURE DIAMOND SPECULAR GLINTS
    # ==========================================================================
    def glint(gx: int, gy: int, r: int = 3) -> None:
        c.set_px(gx, gy, P["BRASS_SPEC"])
        for d in range(1, r + 1):
            col = P["BRASS_SPEC"] if d <= 2 else P["BRASS_GLEAM"]
            c.set_px(gx - d, gy, col)
            c.set_px(gx + d, gy, col)
            c.set_px(gx, gy - d, col)
            c.set_px(gx, gy + d, col)
        if r >= 3:
            # Tiny 1px diagonal flare around main sparkle center
            for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                c.set_px(gx + ox, gy + oy, P["BRASS_GLEAM"])

    glint(280, 39, r=4)   # Hero star glint on top-left of telescope
    glint(317, 80, r=2)   # Glint on right rim of azimuth plate
    glint(320, 154, r=3)  # Glint on upper-right tripod leg band
    glint(296, 194, r=4)  # Hero star glint on center leg mid-clamp
    glint(333, 195, r=2)  # Glint on right leg mid-clamp

    return c



# Muted, Weathered Steel & Ash/Walnut Palette for the High-Density Ice Axe
ICE_AXE_PAL: Dict[str, RGBA] = {
    "INK_DARK":    (18, 14, 24, 255),     # Crisp twilight outline
    "STEEL_OUT":   (28, 28, 38, 255),     # Muted dark gunmetal outline
    "STEEL_DEEP":  (34, 34, 42, 255),     # Deep weathered forged iron
    "STEEL_SHAD":  (56, 56, 66, 255),     # Dark patinated steel
    "STEEL_MID":   (84, 82, 92, 255),     # Matte mountain steel
    "STEEL_LIGHT": (122, 118, 126, 255),  # Worn brushed steel bevel
    "STEEL_GLEAM": (166, 160, 158, 255),  # Subtle worn edge catchlight
    "STEEL_SPEC":  (204, 196, 188, 255),  # Soft pinpoint highlight (muted twilight)
    "STEEL_WARM":  (110, 86, 70, 255),    # Subtle tarnished bronze/dust patina
    "WOOD_OUT":    (32, 20, 22, 255),     # Muted dark umber shaft outline
    "WOOD_DARK":   (38, 24, 24, 255),     # Deep shadowed weathered walnut
    "WOOD_SHAD":   (60, 38, 32, 255),     # Dark seasoned ash/walnut
    "WOOD_MID":    (88, 56, 44, 255),     # Earthy mid hickory
    "WOOD_LIGHT":  (122, 82, 62, 255),    # Desaturated sunlit wood
    "WOOD_RIM":    (150, 108, 82, 255),   # Muted warm rim highlight
}


def render_polished_ice_axe(scale_from_2x: int = 2, sprite_scale: float = 0.86) -> Canvas:
    """
    Renders a weathered, sunset-muted mountaineering Ice Axe directly at `scale_from_2x`
    relative to the 960x540 foreground grid (`scale_from_2x=2` -> 1920x1080 native
    pixel density; `scale_from_2x=1` -> 960x540 for shadow casting & half-res banner),
    preserving its orientation while scaling uniformly around `(102, 294)` by `sprite_scale`.
    """
    S = max(1, scale_from_2x)
    K = sprite_scale
    piv_x, piv_y = 102.0, 294.0

    def _pt(px: float, py: float) -> Tuple[float, float]:
        return ((piv_x + (px - piv_x) * K) * S, (piv_y + (py - piv_y) * K) * S)

    W, H = FG_W * S, FG_H * S
    c = Canvas(W, H)
    P = ICE_AXE_PAL

    steel_ramp: List[RGBA] = [
        P["STEEL_DEEP"], P["STEEL_SHAD"], P["STEEL_MID"], P["STEEL_LIGHT"], P["STEEL_GLEAM"], P["STEEL_SPEC"]
    ]
    wood_ramp: List[RGBA] = [
        P["WOOD_DARK"], P["WOOD_SHAD"], P["WOOD_MID"], P["WOOD_LIGHT"], P["WOOD_RIM"]
    ]

    ax, ay = _pt(143.5, 146.0)
    bx, by = _pt(96.0, 311.5)
    vx, vy = bx - ax, by - ay
    seg_len = math.hypot(vx, vy)
    tx, ty = vx / seg_len, vy / seg_len
    # Unit normal pointing down-right (+x, +y toward the sunset)
    nx, ny = -ty, tx

    def _lerp_col(a: RGBA, b: RGBA, t: float) -> RGBA:
        tt = max(0.0, min(1.0, t))
        return (
            int(round(a[0] + (b[0] - a[0]) * tt)),
            int(round(a[1] + (b[1] - a[1]) * tt)),
            int(round(a[2] + (b[2] - a[2]) * tt)),
            255,
        )

    def _bayer_ramp(px: int, py: int, stops: Sequence[RGBA], t: float) -> RGBA:
        return bayer_pick(px, py, stops, max(0.0, min(1.0, t)))

    # 1. Forged steel head geometry (adze + boss + curved serrated pick) in scaled coords
    adze_top_pts = [_pt(111.5, 138.0), _pt(126.0, 137.5), _pt(140.0, 138.8)]
    adze_bot_pts = [_pt(111.5, 144.2), _pt(117.5, 146.2), _pt(125.0, 145.0), _pt(138.5, 146.5)]
    adze_poly: List[Point] = adze_top_pts + list(reversed(adze_bot_pts))

    boss_poly: List[Point] = [_pt(px, py) for (px, py) in [
        (137.0, 138.2), (148.5, 139.2), (153.0, 142.8),
        (150.5, 149.8), (138.0, 148.5),
    ]]

    pick_top_pts = [_pt(147.0, 139.5), _pt(160.0, 143.0), _pt(173.0, 149.2), _pt(185.5, 158.8)]
    pick_bot_pts = [_pt(148.0, 148.5), _pt(159.0, 150.2), _pt(172.0, 155.2), _pt(185.2, 160.0)]
    pick_poly: List[Point] = pick_top_pts + list(reversed(pick_bot_pts))

    # Track interior kind for selective outline pass: 0=empty, 1=steel, 2=wood, 3=carabiner_hole
    kind = [[0 for _ in range(W)] for _ in range(H)]

    eye_cx, eye_cy = _pt(144.0, 144.2)
    eye_r = 1.65 * K * S

    all_pts = adze_poly + boss_poly + pick_poly + [(ax, ay), (bx, by)]
    y_min = max(1, int(math.floor(min(p[1] for p in all_pts) - 8 * S)))
    y_max = min(H - 2, int(math.ceil(max(p[1] for p in all_pts) + 8 * S)))
    x_min = max(1, int(math.floor(min(p[0] for p in all_pts) - 8 * S)))
    x_max = min(W - 2, int(math.ceil(max(p[0] for p in all_pts) + 8 * S)))

    pick_teeth_x0, _ = _pt(158.5, 145.0)
    pick_teeth_x1, _ = _pt(179.0, 154.0)
    adze_bright_x0, _ = _pt(118.0, 138.0)
    adze_bright_x1, _ = _pt(135.0, 138.0)
    adze_tip_x, _ = _pt(115.0, 140.0)
    boss_x0, boss_y0 = _pt(137.0, 138.2)
    pick_spine_x0, _ = _pt(149.0, 140.0)
    pick_spine_x1, _ = _pt(170.0, 148.0)
    pick_fade_x, _ = _pt(165.0, 145.0)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            px_f, py_f = x + 0.5, y + 0.5
            d_eye = math.hypot(px_f - eye_cx, py_f - eye_cy)
            if d_eye <= eye_r:
                kind[y][x] = 3
                continue

            in_adze = point_in_poly(px_f, py_f, adze_poly)
            in_boss = point_in_poly(px_f, py_f, boss_poly)
            in_pick = point_in_poly(px_f, py_f, pick_poly)

            if in_pick and pick_teeth_x0 <= px_f <= pick_teeth_x1:
                y_bot_pick = interp_y(pick_bot_pts, px_f)
                phase = ((px_f - pick_teeth_x0) / (K * S)) % 4.6
                tooth_cut = max(0.0, 1.0 - abs(phase - 2.3) / 1.45) * (1.35 * K * S)
                if py_f >= y_bot_pick - tooth_cut:
                    in_pick = False

            if in_adze or in_boss or in_pick:
                kind[y][x] = 1
                patina = (value_noise(px_f / S, py_f / S, 3.5, 211) - 0.5) * 0.24 + (fbm_hash(x, y, 307) - 0.5) * 0.14
                if in_adze:
                    y_top_a = interp_y(adze_top_pts, px_f)
                    y_bot_a = interp_y(adze_bot_pts, px_f)
                    v_frac = max(0.0, min(1.0, (py_f - y_top_a) / max(1.0, y_bot_a - y_top_a)))
                    t_a = max(0.0, min(0.86, (1.0 - v_frac) * 0.72 + patina + (0.14 if px_f < adze_tip_x else 0.0)))
                    if v_frac < 0.22 and adze_bright_x0 <= px_f <= adze_bright_x1 and fbm_hash(int(px_f // S), 1, 99) > 0.35:
                        t_a = max(t_a, 0.76)
                    col = _bayer_ramp(x, y, steel_ramp, t_a)
                elif in_boss:
                    u_b = (px_f - boss_x0) / (16.0 * K * S)
                    v_b = (py_f - boss_y0) / (11.6 * K * S)
                    if d_eye <= eye_r + 1.1 * K * S:
                        col = P["STEEL_OUT"]
                    else:
                        t_b = max(0.02, min(0.84, 0.24 + u_b * 0.36 - v_b * 0.26 + patina))
                        col = _bayer_ramp(x, y, steel_ramp, t_b)
                else:
                    y_top_p = interp_y(pick_top_pts, px_f)
                    y_bot_p = interp_y(pick_bot_pts, px_f)
                    v_frac = max(0.0, min(1.0, (py_f - y_top_p) / max(1.0, y_bot_p - y_top_p)))
                    t_p = max(0.0, min(0.88, (1.0 - v_frac) * 0.66 - max(0.0, (px_f - pick_fade_x) / (30.0 * K * S)) * 0.16 + patina))
                    if v_frac < 0.18 and pick_spine_x0 <= px_f <= pick_spine_x1 and fbm_hash(int(px_f // S), 2, 145) > 0.30:
                        t_p = max(t_p, 0.78)
                    col = _bayer_ramp(x, y, steel_ramp, t_p)
                    if 0.35 <= v_frac <= 0.68 and patina > 0.04:
                        col = _lerp_col(col, P["STEEL_WARM"], 0.38)
                c.set_px(x, y, col)
                continue

            # Project pixel onto shaft centerline A -> B
            dx_a, dy_a = px_f - ax, py_f - ay
            proj = dx_a * tx + dy_a * ty
            if proj < 0.0 or proj > seg_len:
                continue
            s = proj / seg_len
            perp = dx_a * nx + dy_a * ny

            if s <= 0.088:
                hw = 5.3 * K * S
            elif s <= 0.815:
                st = (s - 0.088) / (0.815 - 0.088)
                hw = (4.85 - 0.75 * st) * K * S
            elif s <= 0.858:
                hw = 4.35 * K * S
            else:
                st = (s - 0.858) / (1.0 - 0.858)
                hw = (4.05 * (1.0 - st) + 0.55 * st) * K * S

            if abs(perp) > hw:
                continue

            u = perp / max(1.0, hw)

            if s <= 0.088:
                # Weathered patinated steel head collar with worn rivets
                kind[y][x] = 1
                pat = (value_noise(px_f / S, py_f / S, 3.0, 229) - 0.5) * 0.20 + (fbm_hash(x, y, 409) - 0.5) * 0.12
                if s >= 0.080 or s <= 0.008:
                    col = P["STEEL_DEEP"]
                elif abs(s - 0.044) <= 0.011 and (abs(u + 0.35) <= 0.16 or abs(u - 0.42) <= 0.16):
                    col = P["STEEL_LIGHT"] if u > 0 else P["STEEL_MID"]
                else:
                    t_col = max(0.02, min(0.84, 0.36 + u * 0.34 - abs(u - 0.48) * 0.14 + pat))
                    col = _bayer_ramp(x, y, steel_ramp, t_col)
                c.set_px(x, y, col)
            elif s <= 0.815:
                # Seasoned, matte weathered ash/hickory shaft with hand-wear darkening and rock scuffs
                kind[y][x] = 2
                grain = (
                    math.sin(u * 8.0 + value_noise(px_f / S, py_f / S, 4.0, 41) * 2.8) * 0.10
                    + (fbm_hash(int(round(u * 8.0)), int(round(s * 42.0)), 88) - 0.5) * 0.14
                )
                # Matte grip darkening (s ~ 0.16..0.40) and subtle weathered scuffs where shaft leans on rocks (s ~ 0.52..0.76)
                hand_wear = -0.10 * math.exp(-((s - 0.26) / 0.12) ** 2)
                rock_scuff = (
                    -0.18
                    if (0.48 <= s <= 0.78 and value_noise(px_f / S, py_f / S, 2.8, 773) > 0.72)
                    else 0.0
                )
                # Keep warm sunset catchlight narrow and understated along right shoulder
                t_wood = max(
                    0.02,
                    min(0.92, 0.34 + u * 0.42 - max(0.0, u - 0.52) * 0.42 + grain + hand_wear + rock_scuff),
                )
                col = _bayer_ramp(x, y, wood_ramp, t_wood)
                c.set_px(x, y, col)
            else:
                # Weathered lower steel ferrule & rock-scuffed spike cone
                kind[y][x] = 1
                pat_sp = (value_noise(px_f / S, py_f / S, 2.5, 503) - 0.5) * 0.22 + (fbm_hash(x, y, 509) - 0.5) * 0.14
                if 0.854 <= s <= 0.864:
                    col = P["STEEL_DEEP"]
                else:
                    spec_peak = max(0.0, 1.0 - abs(u - 0.42) / 0.28)
                    t_sp = max(0.04, min(0.88, 0.24 + u * 0.28 + spec_peak * 0.36 + pat_sp))
                    col = _bayer_ramp(x, y, steel_ramp, t_sp)
                c.set_px(x, y, col)

    # 2. Clean selective 1px outer border pass ("selout") matching theodolite silhouette quality
    outline_pts: List[Tuple[int, int, RGBA]] = []
    for y in range(max(1, y_min - 2), min(H - 1, y_max + 2)):
        for x in range(max(1, x_min - 2), min(W - 1, x_max + 2)):
            if kind[y][x] not in (0, 3):
                continue
            adj_steel = False
            adj_wood = False
            left_lit = False
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nk = kind[y + dy][x + dx]
                if nk == 1:
                    adj_steel = True
                    if dx < 0 or dy < 0:
                        left_lit = True
                elif nk == 2:
                    adj_wood = True
                    if dx < 0:
                        left_lit = True
            if not (adj_steel or adj_wood):
                continue
            if kind[y][x] == 3:
                # Keep interior of carabiner hole empty except 1px rim if desired (hole center stays transparent)
                continue
            if adj_steel:
                col_out = P["STEEL_OUT"] if left_lit else P["INK_DARK"]
            else:
                col_out = P["WOOD_OUT"] if left_lit else P["INK_DARK"]
            outline_pts.append((x, y, col_out))

    for ox, oy, ocol in outline_pts:
        c.set_px(ox, oy, ocol)

    return c



# Dedicated Sunset-Harmonized Stone Palette for the 2x Shadow-Free Cairn
CAIRN_PAL: Dict[str, RGBA] = {
    "INK":        (18, 15, 24, 255),     # Deep stone crevice & underside outline
    "OUT_SLATE":  (30, 26, 36, 255),     # Selective cool charcoal-umber outline on shadowed left edges
    "OUT_UMBER":  (54, 34, 28, 255),     # Selective warm outline on sunlit right edges
    "SLATE_0":    (34, 30, 42, 255),     # Deepest warm-charcoal rock shadow
    "SLATE_1":    (52, 46, 54, 255),     # Mid charcoal-taupe shadow
    "SLATE_2":    (76, 65, 66, 255),     # Lit weathered taupe-slate stone
    "SLATE_RIM":  (106, 92, 86, 255),    # Top-left twilight sky-bounce bevel
    "OCHRE_0":    (84, 54, 42, 255),     # Deep terracotta shadow bridge
    "OCHRE_1":    (118, 76, 52, 255),    # Warm umber-ochre midtone
    "OCHRE_2":    (154, 102, 64, 255),   # Warm sandstone
    "OCHRE_3":    (190, 134, 80, 255),   # Sunlit alpenglow ochre
    "OCHRE_4":    (222, 168, 106, 255),  # Bright sunlit rim
    "OCHRE_5":    (246, 206, 146, 255),  # Peak alpenglow stone highlight
}


def render_faithful_cairn_sprite_2x(
    grid_fg: List[List[RGB]],
    fg_masks: Dict[str, List[List[bool]]],
    palette: Sequence[RGB],
) -> Canvas:
    """
    Renders a faithful, shadow-free, ice-axe-free 960x540 (2x) Cairn sprite conditioned
    directly on `assets/banner.jpg` (`grid_fg`), preserving the original's disheveled
    stone silhouette, rich colors, lighting, AND its full extended right foot behind the
    theodolite (out to x=278).
    """
    c = Canvas(FG_W, FG_H)

    # 1. Build full 2x cairn mask (x=38..279, y=31..284) preserving Stone 3's rounded top crest behind the tripod leg
    cairn_mask = [[False for _ in range(FG_W)] for _ in range(FG_H)]
    cairn_lower_full_2x: List[Point] = [
        (72, 196), (202, 195), (213, 206), (223, 197), (236, 195), (250, 201), (265, 210),
        (278, 221), (279, 234), (276, 244), (244, 252), (204, 260),
        (134, 270), (96, 282), (42, 281), (38, 260), (54, 224),
    ]
    stone3_leg_arch: List[Point] = [(223, 197), (236, 196), (249, 202), (263, 210), (271, 216), (277, 222)]
    # Gently scalloped boulder-base contour (follows each foot boulder belly instead of a ruler-straight diagonal)
    base_floor_pts: List[Point] = [
        (42, 274), (50, 278), (66, 281), (82, 280), (98, 276), (118, 271),
        (132, 267), (156, 265), (178, 260), (184, 258), (206, 256),
        (224, 252), (230, 250), (254, 248), (270, 243), (278, 237),
    ]

    for y in range(31, 285):
        for x in range(40, 280):
            if y > int(round(interp_y(base_floor_pts, x))):
                continue
            col = grid_fg[y][x]
            r, g, b = col
            lum = rgb_lum(col)
            if y <= 206 and 68 <= x <= 216:
                if lum < 36.0 or r > b + 7:
                    cairn_mask[y][x] = True
            elif y > 192 and point_in_poly(x, y, cairn_lower_full_2x):
                # Clamp Stone 3's curved dome arch cleanly across the tripod leg so leg doesn't protrude above the stone
                if x >= 223 and y < int(round(interp_y(stone3_leg_arch, x))):
                    continue
                # Exclude twilight purple sky above the sloping right foot (x > 206, y < 226)
                if x > 206 and y < 226 and b > r + 2 and lum > 50.0:
                    continue
                # Exclude faint background-slate fringe clinging to the far-left lower flank
                if x <= 56 and y >= 248 and b > r + 4 and lum > 38.0:
                    continue
                # Keep warm stone foot & dark boulder crevices; exclude exposed blue-slate platform
                if not is_platform_slate_or_shadow(col) or (68 <= x <= 204 and y <= 258):
                    cairn_mask[y][x] = True

    # Fill interior stone/shadow gaps without bridging across the sky cleft above Stone 3 (x=202..224, y=194..206)
    fill_scanline_spans(cairn_mask, 32, 281, 40, 200)
    fill_scanline_spans(cairn_mask, 206, 256, 200, 279)
    despike_mask(cairn_mask, min_neighbors=2)

    # 2. Build generous occlusion mask covering:
    #    - ice_axe dilated by 3px + dark cast shadow band around pick & shaft
    #    - theodolite left tripod leg (x=236..272, y=190..265)
    #    - coiled rope overlap on lower cairn foot (y=218..280, x=122..272)
    occ_axe = [[False for _ in range(FG_W)] for _ in range(FG_H)]
    for y in range(132, 300):
        for x in range(82, 196):
            d_shaft = dist_point_to_seg(x, y, 143.0, 146.0, 96.0, 310.0)
            # Dilate around shaft (width 7.8 covers dark outline + shadow) and pick/adze
            if y >= 142 and d_shaft <= 7.8:
                occ_axe[y][x] = True
            else:
                for oy in range(-3, 5):
                    ny = max(0, min(FG_H - 1, y + oy))
                    for ox in range(-3, 4):
                        nx = max(0, min(FG_W - 1, x + ox))
                        if fg_masks["ice_axe"][ny][nx]:
                            occ_axe[y][x] = True
                            break
                    if occ_axe[y][x]:
                        break

    occ_leg = [[False for _ in range(FG_W)] for _ in range(FG_H)]
    for y in range(188, 266):
        for x in range(232, 274):
            if dist_point_to_seg(x, y, 262.0, 192.0, 244.0, 260.0) <= 7.2:
                occ_leg[y][x] = True

    occ_rope = [[False for _ in range(FG_W)] for _ in range(FG_H)]
    for y in range(218, 284):
        for x in range(122, 272):
            if fg_masks["rope"][y][x] or fg_masks["gloves"][y][x]:
                occ_rope[y][x] = True
            else:
                # 2px dilation around rope boundary so rope outline doesn't bleed into cairn
                for oy in (-2, -1, 0, 1, 2):
                    ny = max(0, min(FG_H - 1, y + oy))
                    for ox in (-2, -1, 0, 1, 2):
                        nx = max(0, min(FG_W - 1, x + ox))
                        if fg_masks["rope"][ny][nx]:
                            occ_rope[y][x] = True
                            break
                    if occ_rope[y][x]:
                        break

    def is_clean(px: int, py: int) -> bool:
        if not (0 <= px < FG_W and 0 <= py < FG_H):
            return False
        return cairn_mask[py][px] and not occ_axe[py][px] and not occ_leg[py][px] and not occ_rope[py][px]

    # 3. Reconstruct ONLY occluded areas while preserving 100% of the 3 distinct AI-parsed foot stones:
    #    - Stone 1 (x~86..169): left-center dark base boulder + warm upper-right wedge
    #    - Stone 2 (x~169..217): tall slanted central-right golden-ochre boulder
    #    - Stone 3 (x~217..279): broad rounded outer-right dome boulder behind the tripod leg
    reconstructed: Dict[Tuple[int, int], RGB] = {}

    def _x_seam_12(py: int) -> float:
        return interp_y([(220, 175.0), (236, 171.0), (250, 169.0), (266, 167.0)], py)

    def _x_seam_23(py: int) -> float:
        return interp_y([(202, 198.0), (214, 212.0), (226, 217.0), (240, 218.0), (256, 213.0)], py)

    def _stone_id(px: int, py: int) -> int:
        if px < _x_seam_12(py):
            return 1
        if px < _x_seam_23(py):
            return 2
        return 3

    def _lerp_rgb(a: RGB, b: RGB, t: float) -> RGB:
        tt = max(0.0, min(1.0, t))
        return (
            int(round(a[0] + (b[0] - a[0]) * tt)),
            int(round(a[1] + (b[1] - a[1]) * tt)),
            int(round(a[2] + (b[2] - a[2]) * tt)),
        )

    def _smooth_clean_patch(px: int, py: int, sid: int, radius: int = 2) -> RGB:
        sr = sg = sb = cnt = 0
        for oy in range(-1, 2):
            for ox in range(-radius, radius + 1):
                cx, cy = px + ox, py + oy
                if is_clean(cx, cy) and _stone_id(cx, cy) == sid:
                    r, g, b = grid_fg[cy][cx]
                    sr += r
                    sg += g
                    sb += b
                    cnt += 1
        if cnt == 0:
            return grid_fg[py][px]
        return (sr // cnt, sg // cnt, sb // cnt)

    # First pass: bridge Stone 3 across occ_leg (outside occ_rope) along its curved dome slope (dy/dx ~ +0.40)
    leg_repaired: Dict[Tuple[int, int], RGB] = {}
    for y in range(192, 248):
        for x in range(232, 274):
            if not (cairn_mask[y][x] and occ_leg[y][x] and not occ_rope[y][x]):
                continue
            if not cairn_mask[y - 1][x]:
                leg_repaired[(x, y)] = (38, 26, 28)
                continue
            if not cairn_mask[y - 2][x]:
                jitter = (fbm_hash(x, y, 311) - 0.5) * 16.0
                leg_repaired[(x, y)] = (
                    int(max(0, min(255, 184 + jitter))),
                    int(max(0, min(255, 126 + jitter * 0.7))),
                    int(max(0, min(255, 70 + jitter * 0.4))),
                )
                continue

            lx = rx = None
            ly_val = ry_val = y
            for d in range(1, 32):
                if lx is None:
                    cx = x - d
                    cy = int(round(y - 0.40 * d))
                    if is_clean(cx, cy) and _stone_id(cx, cy) == 3:
                        lx, ly_val = cx, cy
                if rx is None:
                    cx = x + d
                    cy = int(round(y + 0.40 * d))
                    if is_clean(cx, cy) and _stone_id(cx, cy) == 3:
                        rx, ry_val = cx, cy
                if lx is not None and rx is not None:
                    break

            if lx is not None and rx is not None:
                cL = _smooth_clean_patch(lx, ly_val, 3, radius=1)
                cR = _smooth_clean_patch(rx, ry_val, 3, radius=1)
                u = (x - lx) / float(rx - lx)
                grain_u = max(0.0, min(1.0, u + (fbm_hash(x, y, 613) - 0.5) * 0.14))
                leg_repaired[(x, y)] = _lerp_rgb(cL, cR, grain_u)
            elif lx is not None:
                leg_repaired[(x, y)] = _smooth_clean_patch(lx, ly_val, 3, radius=1)
            elif rx is not None:
                leg_repaired[(x, y)] = _smooth_clean_patch(rx, ry_val, 3, radius=1)
            else:
                leg_repaired[(x, y)] = (156, 102, 58)

    def _stone3_or_clean_px(px: int, py: int) -> Optional[RGB]:
        if (px, py) in leg_repaired:
            return leg_repaired[(px, py)]
        if is_clean(px, py):
            return grid_fg[py][px]
        return None

    for y in range(31, 285):
        for x in range(38, 280):
            if not cairn_mask[y][x]:
                continue
            if (x, y) in leg_repaired:
                reconstructed[(x, y)] = leg_repaired[(x, y)]
                continue
            if not (occ_axe[y][x] or occ_leg[y][x] or occ_rope[y][x]):
                reconstructed[(x, y)] = grid_fg[y][x]
                continue

            if occ_rope[y][x]:
                sid = _stone_id(x, y)
                s12 = _x_seam_12(y)
                s23 = _x_seam_23(y)
                # Preserve distinct stone boundaries with crisp 1-2px dark rock crevices between Stone 1 | Stone 2 | Stone 3
                if abs(x - s12) <= 1.05 or abs(x - s23) <= 1.05:
                    reconstructed[(x, y)] = (32, 24, 28)
                    continue
                # Crisp dark sitting edge along the bottom of the foot
                if not cairn_mask[y + 1][x] or not cairn_mask[y + 2][x]:
                    reconstructed[(x, y)] = (34, 26, 30)
                    continue

                # Find the clean/leg-repaired boundary directly above occ_rope for seamless join
                y_join: Optional[int] = None
                dy_src = 0
                for dy in range(1, 46):
                    cy = y - dy
                    if cy < 192 or not cairn_mask[cy][x]:
                        break
                    if _stone_id(x, cy) != sid:
                        continue
                    if _stone3_or_clean_px(x, cy) is not None:
                        y_join = cy
                        dy_src = dy
                        break

                # Smooth horizontally across +/-4px along y_join so zero vertical pinstripes project down!
                join_col: Optional[RGB] = None
                if y_join is not None:
                    jr = jg = jb = jcnt = 0
                    for ox in range(-4, 5):
                        for oy in (-1, 0):
                            sx, sy = x + ox, y_join + oy
                            if _stone_id(sx, sy) == sid:
                                cand = _stone3_or_clean_px(sx, sy)
                                if cand is not None:
                                    jr += cand[0]
                                    jg += cand[1]
                                    jb += cand[2]
                                    jcnt += 1
                    if jcnt > 0:
                        join_col = (jr // jcnt, jg // jcnt, jb // jcnt)

                # Per-stone natural boulder face palette (matches the 3 original AI stones cleanly)
                if sid == 1:
                    stone_mid: RGB = (54, 48, 56) if x < 144 else (92, 68, 56)
                    stone_bot: RGB = (34, 30, 38)
                elif sid == 2:
                    stone_mid = (84, 58, 44) if x < 185 else (154, 102, 58)
                    stone_bot = (48, 36, 34)
                else:
                    stone_mid = (128, 84, 52) if x < 238 else (168, 114, 66)
                    stone_bot = (54, 38, 36)

                if join_col is None:
                    join_col = stone_mid

                # Distance to bottom of cairn_mask at column x for rounded lower boulder curve
                y_bot = y
                while y_bot + 1 < 285 and cairn_mask[y_bot + 1][x]:
                    y_bot += 1
                dist_bot = y_bot - y
                span = max(8.0, float(dy_src + dist_bot))
                v_frac = min(1.0, dy_src / span)

                # Quick smooth hand-off from exact top join (dy_src=1..8px) -> smooth 2D stone body -> rounded bottom shadow
                join_keep = max(0.0, 1.0 - (dy_src - 1.0) / 8.0)
                body_col = _lerp_rgb(stone_mid, join_col, join_keep)
                # Use smooth per-stone top-to-bottom progress independent of occ_leg step height
                y_stone_top = 228.0 if sid == 1 else (204.0 if sid == 2 else 197.0)
                v_frac = max(0.0, min(1.0, (y - y_stone_top) / max(12.0, float(y_bot) - y_stone_top)))
                round_t = max(0.0, min(0.86, (v_frac ** 2.2) * 0.82))
                strata_n = (value_noise(x * 0.45, y * 1.6, 4.5, 521 + sid * 23) - 0.5) * 0.14
                grain_n = (fbm_hash(x, y, 733 + sid * 17) - 0.5) * 0.14
                shaded = _lerp_rgb(body_col, stone_bot, max(0.0, min(0.88, round_t + strata_n + grain_n)))

                # Soft 3D bevel shadow along Stone 1|2 and Stone 2|3 crevices
                near_seam = min(abs(x - s12), abs(x - s23))
                if near_seam < 4.5:
                    shaded = _lerp_rgb(shaded, (36, 28, 32), (4.5 - near_seam) / 4.5 * 0.42)
                reconstructed[(x, y)] = shaded
                continue

            # Behind the ice axe shaft (outside occ_rope): clean un-warped horizontal stone-bed continuation
            lx = rx = None
            for d in range(1, 48):
                if lx is None and is_clean(x - d, y):
                    lx = x - d
                if rx is None and is_clean(x + d, y):
                    rx = x + d
                if lx is not None and rx is not None:
                    break

            if lx is not None and rx is not None:
                u = (x - lx) / float(rx - lx)
                grain_u = max(0.0, min(1.0, u + (fbm_hash(x, y, 909) - 0.5) * 0.18))
                reconstructed[(x, y)] = _lerp_rgb(grid_fg[y][lx], grid_fg[y][rx], grain_u)
            elif lx is not None:
                reconstructed[(x, y)] = grid_fg[y][lx]
            elif rx is not None:
                reconstructed[(x, y)] = grid_fg[y][rx]
            else:
                reconstructed[(x, y)] = (62, 52, 54)

    # 4. Quantize to master 64-color VGA palette with crisp low Bayer dithering
    def _is_sky_zone(px: int, py: int) -> bool:
        if px < 100:
            return py <= 220
        if px < 200:
            return py <= 205
        return py <= 232

    # Clean 1px twilight-tinted fringe nibs along the outermost sky perimeter only (interior untouched)
    to_trim: List[Tuple[int, int]] = []
    for y in range(31, 234):
        for x in range(38, 280):
            if not cairn_mask[y][x] or not _is_sky_zone(x, y):
                continue
            if any(not cairn_mask[y + dy][x + dx] for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                col = reconstructed.get((x, y))
                if col is not None and col[2] >= col[0] - 2 and rgb_lum(col) > 52.0:
                    to_trim.append((x, y))
    for tx, ty in to_trim:
        cairn_mask[ty][tx] = False
        reconstructed.pop((tx, ty), None)

    # Remove single-pixel outer spurs (<= 1 cardinal neighbor) across full cairn perimeter
    spurs: List[Tuple[int, int]] = []
    for y in range(31, 284):
        for x in range(38, 280):
            if not cairn_mask[y][x]:
                continue
            n4 = sum(1 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)) if cairn_mask[y + dy][x + dx])
            if n4 <= 1:
                spurs.append((x, y))
    for sx, sy in spurs:
        cairn_mask[sy][sx] = False
        reconstructed.pop((sx, sy), None)

    for (x, y), col in reconstructed.items():
        c.set_px(x, y, quantize_pixel_bayer(x, y, col, palette, dither_strength=0.22))

    # 5. Apply a clean, uniform 1px dark pixel-art outline along the sky perimeter AND lower boulder base
    sky_border_col: RGBA = (18, 14, 24, 255)
    for y in range(31, 284):
        for x in range(38, 280):
            if not cairn_mask[y][x]:
                continue
            if any(not cairn_mask[y + dy][x + dx] for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                c.set_px(x, y, sky_border_col)

    return c



def render_weathered_gloves_sprite_2x(palette: Optional[Sequence[RGB]] = None) -> Canvas:
    """
    Renders the pair of weathered, muted leather mountaineering gloves directly on
    the 960x540 (2x) foreground grid (`x=184..268, y=279..325`), preserving their
    dull aged-leather character while giving crisp separation to the cuff, thumb,
    splayed front fingers, and tucked back-glove fingers.
    """
    c = Canvas(FG_W, FG_H)

    leather_stops: List[RGBA] = [
        (36, 22, 22, 255),   # deep shadow fold / cuff underside
        (54, 34, 28, 255),   # dark weathered leather
        (78, 50, 40, 255),   # dull mid brown leather
        (104, 68, 52, 255),  # worn ridge leather
        (134, 92, 70, 255),  # muted worn knuckle scuff
    ]
    outline_col: RGBA = (22, 16, 20, 255)
    crease_col: RGBA = (30, 18, 20, 255)

    # Pixel-tag map: 0=empty, 1=back_glove, 2=front_palm_cuff, 3=front_fingers
    gid = [[0 for _ in range(FG_W)] for _ in range(FG_H)]
    fid_map = [[-1 for _ in range(FG_W)] for _ in range(FG_H)]
    val = [[0.0 for _ in range(FG_W)] for _ in range(FG_H)]
    seam_px: set[Tuple[int, int]] = set()

    # 1. Back / underneath glove (upper-right palm + 3 compact padded horizontal fingers)
    back_palm: List[Point] = [(202, 281), (242, 280), (248, 290), (244, 303), (218, 301), (198, 290)]
    back_fingers: List[Tuple[Point, Point, float]] = [
        ((236.0, 283.5), (259.0, 284.5), 2.85),  # top finger
        ((238.0, 289.5), (264.5, 290.8), 2.95),  # middle finger
        ((236.0, 295.8), (258.5, 297.5), 2.80),  # lower finger
    ]

    for y in range(278, 306):
        for x in range(196, 270):
            px, py = x + 0.5, y + 0.5
            in_bp = point_in_poly(px, py, back_palm)
            best_f = -1
            best_d = 99.0
            for fi, (fa, fb, fr) in enumerate(back_fingers):
                d = dist_point_to_seg(px, py, fa[0], fa[1], fb[0], fb[1])
                if d <= fr and d < best_d:
                    best_d = d
                    best_f = fi
            if not (in_bp or best_f >= 0):
                continue
            gid[y][x] = 1
            grain = (fbm_hash(x, y, 503) - 0.5) * 0.14 + (value_noise(px, py, 3.0, 509) - 0.5) * 0.14
            if best_f >= 0:
                fid_map[y][x] = best_f
                fa, fb, fr = back_fingers[best_f]
                dy_norm = (py - (fa[1] + (fb[1] - fa[1]) * max(0.0, min(1.0, (px - fa[0]) / max(1.0, fb[0] - fa[0]))))) / fr
                v = 0.44 - dy_norm * 0.34 - (px - 236.0) / 70.0 * 0.12 + grain
            else:
                v = 0.36 - (py - 280.0) / 28.0 * 0.25 + grain
            val[y][x] = max(0.04, min(0.84, v))

    # 2. Front / top glove: rounded weathered leather cuff + knuckle palm + 4 chunky splayed fingers + thumb
    front_cuff_palm: List[Point] = [
        (185, 291), (192, 283), (212, 281), (229, 285), (240, 294),
        (235, 308), (222, 317), (214, 314), (203, 306), (190, 300), (185, 296),
    ]
    front_fingers: List[Tuple[Point, Point, float]] = [
        ((224.0, 291.5), (239.0, 295.0), 2.75),  # tucked thumb
        ((229.0, 298.5), (243.5, 307.0), 2.95),  # index finger
        ((223.5, 303.5), (238.5, 313.5), 2.95),  # middle finger
        ((218.0, 308.5), (232.0, 318.0), 2.85),  # ring finger
        ((215.0, 312.5), (225.0, 319.5), 2.45),  # pinky finger (aligned smoothly with outer palm)
    ]

    for y in range(279, 326):
        for x in range(183, 254):
            px, py = x + 0.5, y + 0.5
            in_fp = point_in_poly(px, py, front_cuff_palm)
            best_f = -1
            best_d = 99.0
            for fi, (fa, fb, fr) in enumerate(front_fingers):
                d = dist_point_to_seg(px, py, fa[0], fa[1], fb[0], fb[1])
                if d <= fr and d < best_d:
                    best_d = d
                    best_f = fi
            if not (in_fp or best_f >= 0):
                continue

            grain = (fbm_hash(x, y, 601) - 0.5) * 0.14 + (value_noise(px, py, 3.2, 607) - 0.5) * 0.16
            if best_f >= 0:
                fa, fb, fr = front_fingers[best_f]
                vx, vy = fb[0] - fa[0], fb[1] - fa[1]
                slen = math.hypot(vx, vy)
                tx, ty = vx / slen, vy / slen
                nx, ny = ty, -tx
                proj = ((px - fa[0]) * tx + (py - fa[1]) * ty) / slen
                across = ((px - fa[0]) * nx + (py - fa[1]) * ny) / fr
                gid[y][x] = 3
                fid_map[y][x] = 10 + best_f
                v = 0.46 + across * 0.36 - proj * 0.18 + grain
                val[y][x] = max(0.06, min(0.88, v))
            else:
                gid[y][x] = 2
                # Soft leather wrist-cuff fold shading (no harsh dark line slicing across the glove)
                cuff_seam_x = 200.0 + (py - 282.0) * 0.45
                cuff_fold = -0.16 * math.exp(-((px - cuff_seam_x) / 1.9) ** 2)
                knuckle_bump = 0.34 * math.exp(-(((px - 222.0) / 11.0) ** 2 + ((py - 294.0) / 6.5) ** 2))
                cuff_rim = 0.12 if abs(px - (cuff_seam_x - 3.0)) <= 2.2 and py < 296.0 else 0.0
                shade_bot = -0.28 * max(0.0, (py - 295.0) / 14.0)
                v = 0.38 + knuckle_bump + cuff_rim + cuff_fold + shade_bot + grain
                val[y][x] = max(0.05, min(0.95, v))

    # 3. Render muted leather colors + crisp 1px inter-finger seams + uniform 1px outer outline
    for y in range(278, 327):
        for x in range(182, 271):
            g = gid[y][x]
            if g == 0:
                continue
            is_outer = any(gid[y + dy][x + dx] == 0 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            if is_outer:
                c.set_px(x, y, outline_col)
                continue
            if g in (2, 3) and any(gid[y + dy][x + dx] == 1 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                c.set_px(x, y, crease_col)
                continue
            # Draw crisp 1px inter-finger crease wherever two different finger IDs meet
            fi_here = fid_map[y][x]
            if fi_here >= 0 and any(fid_map[y + dy][x + dx] >= 0 and fid_map[y + dy][x + dx] != fi_here for dx, dy in ((0, 1), (1, 0))):
                c.set_px(x, y, crease_col)
                continue
            if (x, y) in seam_px:
                c.set_px(x, y, crease_col)
                continue
            c.set_px(x, y, bayer_pick(x, y, leather_stops, val[y][x]))

    return c


def render_clean_notebook_sprite_2x() -> Canvas:
    """
    Renders a crisp, isometric surveyor's field notebook & pen directly on the
    960x540 (2x) foreground grid (`x=305..376, y=274..309`), preserving the weathered
    parchment palette and pose of banner.jpg while cleanly sharpening page edges,
    bound page block & leather cover, handwritten field ink marks, and the pen.
    """
    c = Canvas(FG_W, FG_H)

    # Isometric top leaf corners (left, top, right, bottom)
    p_L = (307.0, 290.0)
    p_T = (343.0, 275.0)
    p_R = (373.0, 287.0)
    p_B = (336.0, 303.0)
    top_leaf: List[Point] = [p_L, p_T, p_R, p_B]

    # Thick stacked page block (front-left & front-right bound faces, 4px deep)
    block_left: List[Point] = [p_L, p_B, (336.0, 307.5), (307.0, 294.5)]
    block_right: List[Point] = [p_B, p_R, (373.0, 291.5), (336.0, 307.5)]

    paper_stops: List[RGBA] = [
        (194, 158, 122, 255),  # shaded upper-left parchment
        (222, 186, 146, 255),  # warm peach-cream paper
        (238, 208, 162, 255),  # sunlit parchment
        (248, 226, 178, 255),  # bright sunlit ivory highlight
    ]
    edge_paper_left: RGBA = (146, 114, 88, 255)
    edge_paper_right: RGBA = (182, 148, 112, 255)
    cover_dark: RGBA = (48, 30, 28, 255)
    ink_faint: RGBA = (162, 126, 94, 255)
    ink_note: RGBA = (126, 94, 72, 255)
    page_pen_shadow: RGBA = (114, 86, 68, 255)
    outline_dark: RGBA = (22, 16, 20, 255)
    outline_warm: RGBA = (52, 34, 28, 255)

    # Kind map: 0=empty, 1=top_page, 2=block_left, 3=block_right
    kind = [[0 for _ in range(FG_W)] for _ in range(FG_H)]

    # 1. Rasterize top parchment leaf & subtle sepia contour sketch / ruled lines
    # Basis directions along page axes: u along p_L->p_T (back-right), v along p_L->p_B (front-right)
    ux, uy = p_T[0] - p_L[0], p_T[1] - p_L[1]
    vx, vy = p_B[0] - p_L[0], p_B[1] - p_L[1]
    det = ux * vy - uy * vx

    for y in range(273, 310):
        for x in range(305, 376):
            px, py = x + 0.5, y + 0.5
            if point_in_poly(px, py, top_leaf):
                kind[y][x] = 1
                dx, dy = px - p_L[0], py - p_L[1]
                u = (dx * vy - dy * vx) / det   # 0 (left/front-left) .. 1 (back-right)
                v = (ux * dy - uy * dx) / det   # 0 (back-left) .. 1 (front-right)
                # Warm low-right sunlit shading across the open page + subtle corner paper curl
                grain = (fbm_hash(x, y, 701) - 0.5) * 0.10 + (value_noise(px, py, 5.0, 709) - 0.5) * 0.08
                t_p = max(0.05, min(0.98, 0.35 + v * 0.42 + u * 0.14 - (1.0 - v) * 0.10 + grain))
                col = bayer_pick(x, y, paper_stops, t_p)

                # Subtle sepia notebook rules & tiny field-note ink marks
                if 0.12 <= u <= 0.88 and 0.14 <= v <= 0.88:
                    # Faint ruled page lines parallel to top-left/bottom-right edge
                    line_phase = (v * 6.5) % 1.0
                    if line_phase < 0.13:
                        col = ink_faint
                    # Tiny broken handwritten notes / survey tick marks
                    if (0.20 <= u <= 0.66 and 0.18 <= v <= 0.40) or (0.34 <= u <= 0.78 and 0.62 <= v <= 0.80):
                        if line_phase < 0.24 and fbm_hash(int(u * 26), int(v * 20), 719) > 0.42:
                            col = ink_note
                c.set_px(x, y, col)
            elif point_in_poly(px, py, block_left):
                kind[y][x] = 2
                y_top = interp_y([p_L, p_B], px)
                depth = py - y_top
                if depth >= 3.2:
                    c.set_px(x, y, cover_dark)
                elif int(round(depth)) == 1:
                    c.set_px(x, y, (118, 90, 70, 255))
                else:
                    c.set_px(x, y, edge_paper_left)
            elif point_in_poly(px, py, block_right):
                kind[y][x] = 3
                y_top = interp_y([p_B, p_R], px)
                depth = py - y_top
                if depth >= 3.2:
                    c.set_px(x, y, (66, 44, 36, 255))
                elif int(round(depth)) == 1:
                    c.set_px(x, y, (148, 118, 88, 255))
                else:
                    c.set_px(x, y, edge_paper_right)

    # 2. Clean 1px selective perimeter outline on the notebook body BEFORE drawing the pen
    nb_out: List[Tuple[int, int, RGBA]] = []
    for y in range(273, 310):
        for x in range(305, 376):
            if kind[y][x] != 0:
                continue
            adj_top = any(kind[y + dy][x + dx] == 1 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            adj_bot = any(kind[y + dy][x + dx] in (2, 3) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            if not (adj_top or adj_bot):
                continue
            nb_out.append((x, y, outline_warm if (adj_top and not adj_bot and x >= 338) else outline_dark))
    for ox, oy, ocol in nb_out:
        c.set_px(ox, oy, ocol)
        kind[oy][ox] = 4

    # 3. Crisp dark-ebony & brass-trimmed surveyor's pen resting across the page ((316, 291) -> (361, 283))
    pen_tip = (316.0, 291.2)
    pen_tail = (361.5, 283.0)
    pvx, pvy = pen_tail[0] - pen_tip[0], pen_tail[1] - pen_tip[1]
    plen = math.hypot(pvx, pvy)
    ptx, pty = pvx / plen, pvy / plen
    # Normal pointing up-left across barrel (-pty, ptx -> upward is negative y!)
    pnx, pny = pty, -ptx  # (~ -0.18, -0.98 -> points UP across barrel)

    pen_stops: List[RGBA] = [
        (28, 18, 20, 255),   # dark espresso barrel bottom
        (56, 38, 34, 255),   # mid mahogany-ebony barrel
        (92, 66, 54, 255),   # lit barrel upper bevel
        (134, 104, 84, 255), # subtle glossy catchlight
    ]
    brass_pen_col: RGBA = (182, 134, 68, 255)
    brass_pen_hi: RGBA = (224, 178, 98, 255)
    nib_steel: RGBA = (176, 174, 178, 255)

    for y in range(278, 298):
        for x in range(313, 366):
            px, py = x + 0.5, y + 0.5
            dx, dy = px - pen_tip[0], py - pen_tip[1]
            proj = dx * ptx + dy * pty
            perp = dx * pnx + dy * pny  # + is TOP of barrel, - is BOTTOM of barrel

            # Contact shadow directly underneath pen on the paper page
            if 0.0 <= proj <= plen and -3.3 <= perp <= -1.4 and kind[y][x] == 1:
                c.set_px(x, y, page_pen_shadow)

            if proj < -0.5 or proj > plen + 0.8:
                continue
            s = proj / plen
            # Conical nib tip (s < 0.14), sleek cylinder (0.14..0.95), rounded cap end (>0.95)
            if s < 0.14:
                hw = 0.45 + (max(0.0, s) / 0.14) * 1.45
            elif s > 0.95:
                hw = 1.90 * max(0.25, 1.0 - (s - 0.95) / 0.08)
            else:
                hw = 1.95

            if abs(perp) > hw + 0.85:
                continue
            if abs(perp) > hw:
                # Tight 1px pen outline separating it from bright paper
                c.set_px(x, y, outline_dark)
                continue

            u_p = perp / max(0.8, hw)  # -1 bottom .. +1 top
            if s < 0.11:
                c.set_px(x, y, nib_steel if u_p > -0.1 else (96, 92, 98, 255))
            elif 0.11 <= s <= 0.16 or 0.62 <= s <= 0.66:
                c.set_px(x, y, brass_pen_hi if u_p > 0.15 else brass_pen_col)
            elif 0.66 < s <= 0.93 and u_p >= 0.35:
                # Brass pocket clip along upper cap ridge
                c.set_px(x, y, brass_pen_hi if u_p > 0.55 else brass_pen_col)
            else:
                t_pen = max(0.04, min(0.95, 0.42 + u_p * 0.46))
                c.set_px(x, y, bayer_pick(x, y, pen_stops, t_pen))

    return c


def render_alpine_hut_sprite_2x(hut_scale: float = 0.76) -> Canvas:
    """
    Renders a crisp, unmistakable 3/4 isometric Alpine Refuge Hut on the 960x540 (2x)
    grid, scaled uniformly by `hut_scale` around its bench foundation `(691.5, 412.0)`
    so it matches the compact mountain-refuge scale (`~22x21px` at `0.76`) with sharp
    `1px` outlines, pitched slate roof, stone chimney, door, and subdued framed windows.
    """
    c = Canvas(FG_W, FG_H)
    piv_x, piv_y = 691.5, 412.0

    def _hpt(px: float, py: float) -> Point:
        return (piv_x + (px - piv_x) * hut_scale, piv_y + (py - piv_y) * hut_scale)

    outline_col: RGBA = (20, 22, 32, 255)
    roof_shad: List[RGBA] = [(44, 50, 72, 255), (66, 72, 96, 255)]
    roof_lit: List[RGBA] = [(126, 122, 134, 255), (156, 146, 150, 255), (186, 168, 160, 255)]
    eave_shad: RGBA = (24, 20, 32, 255)
    wall_left_cols: List[RGBA] = [(34, 40, 58, 255), (50, 56, 76, 255)]
    wall_right_cols: List[RGBA] = [(88, 60, 48, 255), (118, 82, 60, 255), (142, 102, 72, 255)]
    base_stone: List[RGBA] = [(34, 40, 48, 255), (52, 56, 64, 255)]
    win_frame: RGBA = (128, 114, 106, 255)
    win_pane: RGBA = (24, 24, 34, 255)
    win_dim: RGBA = (68, 54, 46, 255)
    door_col: RGBA = (22, 18, 26, 255)

    # Interior kind tag: 0=empty, 1=chimney, 2=left_wall, 3=right_wall, 4=roof_left, 5=roof_right
    kind = [[0 for _ in range(FG_W)] for _ in range(FG_H)]
    px_col: Dict[Tuple[int, int], RGBA] = {}

    left_wall_poly: List[Point] = [_hpt(x, y) for (x, y) in [(680, 400), (684.5, 393), (689.5, 400), (689.5, 412), (680, 412)]]
    right_wall_poly: List[Point] = [_hpt(x, y) for (x, y) in [(689.5, 400), (703, 400), (703, 412), (689.5, 412)]]
    roof_left_poly: List[Point] = [_hpt(x, y) for (x, y) in [(678, 400.5), (684.5, 391), (687, 391), (690, 400.5)]]
    roof_right_poly: List[Point] = [_hpt(x, y) for (x, y) in [(685, 391), (699, 391.5), (705, 400.5), (689.5, 400.5)]]
    chimney_poly: List[Point] = [_hpt(x, y) for (x, y) in [(687, 387), (691, 387), (691, 392.5), (687, 392.5)]]

    r_x0, r_y0 = _hpt(685.0, 391.0)
    r_w = 20.0 * hut_scale
    r_h = 9.5 * hut_scale
    seam_x = piv_x + (689.5 - piv_x) * hut_scale
    barge_x, barge_y = _hpt(685.5, 394.5)
    roof_left_mid_x = piv_x + (682.5 - piv_x) * hut_scale
    chim_top_y = int(round(_hpt(689.0, 387.0)[1]))
    chim_lit_x = piv_x + (689.0 - piv_x) * hut_scale
    base_y = piv_y + (410.0 - piv_y) * hut_scale
    eave_y = piv_y + (401.0 - piv_y) * hut_scale
    lw_eave_x = piv_x + (681.5 - piv_x) * hut_scale
    lw_mid_x = piv_x + (685.0 - piv_x) * hut_scale
    lw_bot_y = piv_y + (409.0 - piv_y) * hut_scale
    rw_mid_x = piv_x + (696.0 - piv_x) * hut_scale

    for y in range(385, 416):
        for x in range(676, 708):
            px, py = x + 0.5, y + 0.5
            if point_in_poly(px, py, roof_right_poly):
                kind[y][x] = 5
                u = (px - r_x0) / max(1.0, r_w)
                v = (py - r_y0) / max(1.0, r_h)
                shingle = ((x // 2 + y) & 1) * 0.14
                if v >= 0.86:
                    px_col[(x, y)] = eave_shad
                else:
                    t = max(0.05, min(0.98, 0.35 + u * 0.45 + (1.0 - v) * 0.25 - shingle))
                    px_col[(x, y)] = bayer_pick(x, y, roof_lit, t)
            elif point_in_poly(px, py, roof_left_poly):
                kind[y][x] = 4
                v = (py - r_y0) / max(1.0, r_h)
                if px <= barge_x and py <= barge_y:
                    px_col[(x, y)] = (146, 140, 150, 255)
                elif v >= 0.86:
                    px_col[(x, y)] = eave_shad
                else:
                    px_col[(x, y)] = roof_shad[1] if (px > roof_left_mid_x and v < 0.65) else roof_shad[0]
            elif point_in_poly(px, py, chimney_poly):
                kind[y][x] = 1
                px_col[(x, y)] = (136, 120, 116, 255) if (y <= chim_top_y or px >= chim_lit_x) else (62, 64, 82, 255)
            elif point_in_poly(px, py, left_wall_poly):
                kind[y][x] = 2
                if py >= base_y:
                    px_col[(x, y)] = base_stone[0] if px < lw_mid_x else base_stone[1]
                elif py <= eave_y and px >= lw_eave_x:
                    px_col[(x, y)] = eave_shad
                else:
                    px_col[(x, y)] = wall_left_cols[1] if (px >= lw_mid_x and py < lw_bot_y) else wall_left_cols[0]
            elif point_in_poly(px, py, right_wall_poly):
                kind[y][x] = 3
                if py >= base_y:
                    px_col[(x, y)] = base_stone[1] if px >= rw_mid_x else base_stone[0]
                elif py <= eave_y:
                    px_col[(x, y)] = eave_shad
                else:
                    timber_grain = (fbm_hash(x, y, 811) - 0.5) * 0.25
                    t_w = max(0.05, min(0.95, 0.35 + (px - seam_x) / max(1.0, 14.0 * hut_scale) * 0.45 + timber_grain))
                    px_col[(x, y)] = bayer_pick(x, y, wall_right_cols, t_w)

    # Compact scaled architectural focal details: attic window, door, & two dark-paned framed windows
    ax, ay = _hpt(684.5, 397.5)
    px_col[(int(round(ax)), int(round(ay)))] = (96, 98, 118, 255)

    dx0, dy0 = _hpt(683.2, 405.0)
    dx1, dy1 = _hpt(685.5, 411.8)
    for dy in range(int(round(dy0)), int(round(dy1))):
        for dx in range(int(round(dx0)), int(round(dx1)) + 1):
            if kind[dy][dx] == 2:
                px_col[(dx, dy)] = door_col

    # Two compact 3x3 framed windows on the sunlit right wall
    for wx_ref in (693.0, 698.2):
        wcx, wcy = _hpt(wx_ref, 405.5)
        ix, iy = int(round(wcx)), int(round(wcy))
        for vy in range(iy - 1, iy + 2):
            for vx in range(ix - 1, ix + 2):
                if kind[vy][vx] != 3:
                    continue
                if vx == ix and vy == iy:
                    px_col[(vx, vy)] = win_dim if (wx_ref > 695.0) else win_pane
                else:
                    px_col[(vx, vy)] = win_frame

    for (x, y), col in px_col.items():
        c.set_px(x, y, col)

    # Clean 1px perimeter outline separating the refuge hut sharply from the hillside
    for y in range(385, 416):
        for x in range(676, 708):
            if kind[y][x] != 0:
                continue
            if any(kind[y + dy][x + dx] != 0 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                c.set_px(x, y, outline_col)

    return c


def _render_cobble_cluster_2x(
    stone_specs: Sequence[dict],
    x_bounds: Tuple[int, int],
    y_bounds: Tuple[int, int],
    outline_col: RGBA = (20, 16, 24, 255),
    crevice_col: RGBA = (28, 22, 28, 255),
) -> Canvas:
    """
    Shared 960x540 (2x) cobble rasterizer: fills faceted polygons with 4x4 Bayer-dithered
    granite/ochre stone ramps lit from horizon right (+x, upper-right), marks internal
    stone seams, and traces a 1px exterior contour.
    """
    c = Canvas(FG_W, FG_H)

    def _mix_rgba(a: RGB, b: RGB, t: float) -> RGBA:
        tt = max(0.0, min(1.0, t))
        return (
            int(round(a[0] + (b[0] - a[0]) * tt)),
            int(round(a[1] + (b[1] - a[1]) * tt)),
            int(round(a[2] + (b[2] - a[2]) * tt)),
            255,
        )

    x0, x1 = x_bounds
    y0, y1 = y_bounds
    stone_id_grid = [[-1 for _ in range(FG_W)] for _ in range(FG_H)]
    for sid, spec in enumerate(stone_specs):
        poly = spec["poly"]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        for y in range(min(ys), max(ys) + 1):
            for x in range(min(xs), max(xs) + 1):
                if not point_in_poly(x + 0.5, y + 0.5, poly):
                    continue
                stone_id_grid[y][x] = sid
                nx = (x - spec["cx"]) / spec["rx"]
                ny = (y - spec["cy"]) / spec["ry"]
                # Directional morning sun from top-right + facet variation
                sun_dir = 0.54 + 0.42 * nx - 0.34 * ny
                spot = max(0.0, 1.0 - math.hypot((x - spec["lx"]) / (spec["rx"] * 0.72), (y - spec["ly"]) / (spec["ry"] * 0.78)))
                grain = (fbm_hash(x, y, spec["seed"]) - 0.5) * 0.16 + (fbm_hash(x // 2, y // 2, spec["seed"] + 9) - 0.5) * 0.12
                light = max(0.0, min(1.0, sun_dir + 0.28 * spot + grain))
                # Quantize with 4x4 Bayer dithering across a 5-stop VGA stone ramp
                ramp: Tuple[RGBA, ...] = (
                    _mix_rgba(spec["dark"], spec["dark"], 0.0),
                    _mix_rgba(spec["dark"], spec["mid"], 0.65),
                    _mix_rgba(spec["mid"], spec["warm"], 0.35),
                    _mix_rgba(spec["mid"], spec["warm"], 0.90),
                    _mix_rgba(spec["warm"], spec["hi"], min(1.0, 0.35 + spot * 0.75)),
                )
                c.set_px(x, y, bayer_pick(x, y, ramp, light))

    # Crisp internal rock crevices where stones interlock
    for y in range(y0, y1):
        for x in range(x0, x1):
            sid = stone_id_grid[y][x]
            if sid < 0:
                continue
            for ox, oy in ((1, 0), (0, 1)):
                nsid = stone_id_grid[y + oy][x + ox]
                if nsid >= 0 and nsid != sid:
                    c.set_px(x, y, crevice_col)
                    break

    # Crisp 1px exterior outline
    solid = [[c.get_px(x, y)[3] > 0 for x in range(x0, x1)] for y in range(y0, y1)]
    for dy, y in enumerate(range(y0, y1)):
        for dx, x in enumerate(range(x0, x1)):
            if solid[dy][dx]:
                continue
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = dy + oy, dx + ox
                if 0 <= ny < len(solid) and 0 <= nx < len(solid[0]) and solid[ny][nx]:
                    c.set_px(x, y, outline_col)
                    break

    return c


def render_right_cobbles_sprite_2x() -> Canvas:
    """
    Shadow-free 960x540 (2x) foreground sprite for the 3 interlocking granite/ochre
    cobble stones on the right cliff brink (extracted from the 1x cliff terrain layer).
    Sunlit from horizon right (+x, upper-right) matching the main scene lighting.
    """
    stone_specs = [
        # Back-right chock stone
        {
            "poly": [(420, 254), (436, 253), (443, 263), (430, 268), (418, 263)],
            "dark": (54, 48, 54),
            "mid": (106, 92, 86),
            "warm": (166, 140, 114),
            "hi": (208, 184, 148),
            "cx": 431.0,
            "cy": 260.0,
            "lx": 435.0,
            "ly": 255.0,
            "rx": 14.0,
            "ry": 9.0,
            "seed": 611,
        },
        # Main prominent sunset-lit boulder
        {
            "poly": [(383, 244), (392, 239), (406, 241), (419, 249), (424, 259), (408, 264), (391, 261), (383, 253)],
            "dark": (54, 48, 56),
            "mid": (118, 92, 82),
            "warm": (184, 134, 92),
            "hi": (228, 198, 154),
            "cx": 403.0,
            "cy": 252.0,
            "lx": 413.0,
            "ly": 249.0,
            "rx": 22.0,
            "ry": 14.0,
            "seed": 623,
        },
        # Front-left embedded lip stone
        {
            "poly": [(390, 264), (407, 260), (413, 270), (396, 274), (388, 269)],
            "dark": (48, 42, 48),
            "mid": (86, 74, 70),
            "warm": (158, 130, 102),
            "hi": (214, 188, 152),
            "cx": 400.0,
            "cy": 267.0,
            "lx": 402.0,
            "ly": 261.0,
            "rx": 14.0,
            "ry": 8.0,
            "seed": 637,
        },
    ]
    return _render_cobble_cluster_2x(stone_specs, (381, 446), (237, 276))


def render_platform_cobbles_sprite_2x() -> Canvas:
    """
    Subtle, low-profile weathered granite cobbles and pebbles strewn naturally across
    unobstructed areas of the stone platform, with color temperature warming from cool
    slate on the left to sunlit taupe-ochre toward the eastern rim.
    """
    stone_specs = [
        # 1. Front-left plateau (below cairn boulders, left of ice axe shaft)
        {
            "poly": [(43, 295), (50, 291), (61, 292), (65, 299), (55, 303), (44, 300)],
            "dark": (48, 44, 54),
            "mid": (82, 76, 84),
            "warm": (124, 110, 104),
            "hi": (162, 144, 132),
            "cx": 54.0,
            "cy": 297.0,
            "lx": 59.0,
            "ly": 294.0,
            "rx": 12.0,
            "ry": 7.0,
            "seed": 701,
        },
        {
            "poly": [(66, 298), (73, 297), (76, 302), (69, 305), (65, 302)],
            "dark": (46, 42, 52),
            "mid": (78, 72, 80),
            "warm": (118, 106, 100),
            "hi": (154, 138, 126),
            "cx": 71.0,
            "cy": 301.0,
            "lx": 73.0,
            "ly": 299.0,
            "rx": 7.0,
            "ry": 5.0,
            "seed": 709,
        },
        # 2. Lower-left stray angular chip (below ice axe tip shadow, near front lip)
        {
            "poly": [(85, 323), (92, 321), (97, 325), (91, 329), (84, 327)],
            "dark": (48, 44, 54),
            "mid": (84, 77, 84),
            "warm": (126, 112, 106),
            "hi": (166, 148, 134),
            "cx": 90.5,
            "cy": 325.0,
            "lx": 93.0,
            "ly": 323.0,
            "rx": 7.5,
            "ry": 4.5,
            "seed": 713,
        },
        # 3. Center-front apron (clear between ice axe ferrule and leather gloves)
        {
            "poly": [(132, 325), (141, 322), (153, 324), (156, 330), (144, 333), (133, 330)],
            "dark": (50, 45, 55),
            "mid": (88, 80, 86),
            "warm": (134, 118, 108),
            "hi": (176, 154, 138),
            "cx": 144.0,
            "cy": 328.0,
            "lx": 149.0,
            "ly": 325.0,
            "rx": 13.0,
            "ry": 7.0,
            "seed": 719,
        },
        {
            "poly": [(155, 329), (162, 328), (165, 333), (158, 335), (154, 332)],
            "dark": (48, 44, 52),
            "mid": (84, 76, 82),
            "warm": (128, 114, 104),
            "hi": (168, 146, 132),
            "cx": 160.0,
            "cy": 332.0,
            "lx": 162.0,
            "ly": 330.0,
            "rx": 7.0,
            "ry": 4.5,
            "seed": 727,
        },
        # 4. Front-center-right open granite triangle (below gloves & left of center tripod foot)
        {
            "poly": [(246, 338), (255, 334), (266, 336), (269, 343), (257, 346), (245, 343)],
            "dark": (52, 46, 56),
            "mid": (92, 83, 86),
            "warm": (142, 122, 108),
            "hi": (184, 160, 138),
            "cx": 257.0,
            "cy": 340.0,
            "lx": 262.0,
            "ly": 337.0,
            "rx": 13.0,
            "ry": 7.0,
            "seed": 733,
        },
        {
            "poly": [(232, 344), (239, 342), (243, 347), (236, 349), (231, 347)],
            "dark": (50, 44, 54),
            "mid": (86, 78, 84),
            "warm": (132, 116, 104),
            "hi": (172, 150, 132),
            "cx": 237.0,
            "cy": 345.5,
            "lx": 239.5,
            "ly": 344.0,
            "rx": 7.0,
            "ry": 4.5,
            "seed": 739,
        },
        {
            "poly": [(276, 332), (282, 330), (286, 334), (280, 337), (275, 335)],
            "dark": (52, 46, 54),
            "mid": (94, 84, 86),
            "warm": (144, 124, 108),
            "hi": (184, 160, 138),
            "cx": 280.5,
            "cy": 333.5,
            "lx": 283.0,
            "ly": 332.0,
            "rx": 6.5,
            "ry": 4.0,
            "seed": 743,
        },
        # 5. Front-right brink apron (clear below center tripod foot and left-front of notebook)
        {
            "poly": [(301, 321), (308, 318), (315, 321), (313, 326), (304, 326), (300, 324)],
            "dark": (52, 46, 56),
            "mid": (96, 86, 86),
            "warm": (148, 126, 108),
            "hi": (190, 166, 140),
            "cx": 307.5,
            "cy": 322.5,
            "lx": 311.0,
            "ly": 320.5,
            "rx": 8.5,
            "ry": 5.0,
            "seed": 751,
        },
        {
            "poly": [(316, 324), (321, 323), (323, 327), (318, 328), (315, 326)],
            "dark": (52, 46, 54),
            "mid": (94, 84, 84),
            "warm": (144, 124, 106),
            "hi": (184, 160, 136),
            "cx": 319.0,
            "cy": 325.5,
            "lx": 320.5,
            "ly": 324.5,
            "rx": 5.0,
            "ry": 3.5,
            "seed": 757,
        },
        # 6. Mid-back bench framed cleanly between center & right tripod legs
        {
            "poly": [(310, 254), (319, 250), (328, 252), (330, 258), (320, 260), (310, 258)],
            "dark": (54, 48, 56),
            "mid": (100, 88, 86),
            "warm": (154, 128, 106),
            "hi": (196, 168, 138),
            "cx": 320.0,
            "cy": 255.0,
            "lx": 324.5,
            "ly": 253.0,
            "rx": 11.0,
            "ry": 6.0,
            "seed": 761,
        },
        {
            "poly": [(332, 256), (338, 255), (340, 259), (335, 261), (331, 259)],
            "dark": (52, 46, 54),
            "mid": (96, 86, 84),
            "warm": (148, 124, 104),
            "hi": (190, 164, 136),
            "cx": 335.5,
            "cy": 258.0,
            "lx": 337.5,
            "ly": 256.5,
            "rx": 5.5,
            "ry": 3.5,
            "seed": 769,
        },
        # 7. Stray sunlit chip rolled slightly left of the main right-brink cobbles
        {
            "poly": [(362, 255), (369, 252), (375, 255), (372, 260), (364, 260), (361, 258)],
            "dark": (54, 48, 56),
            "mid": (104, 90, 86),
            "warm": (160, 134, 110),
            "hi": (202, 176, 144),
            "cx": 368.0,
            "cy": 256.5,
            "lx": 371.5,
            "ly": 254.5,
            "rx": 8.0,
            "ry": 4.5,
            "seed": 773,
        },
    ]
    return _render_cobble_cluster_2x(
        stone_specs,
        (40, 380),
        (248, 353),
        outline_col=(36, 32, 42, 255),
        crevice_col=(34, 30, 40, 255),
    )


def apply_composite_shadows_2x(
    scene_960: Canvas,
    s_cairn: Canvas,
    s_rope: Canvas,
    s_ice_axe: Canvas,
    s_gloves: Canvas,
    s_notebook: Canvas,
    s_theodolite: Canvas,
    s_cobbles: Optional[Canvas] = None,
    s_platform_cobbles: Optional[Canvas] = None,
) -> Tuple[Canvas, Canvas, Canvas, Canvas]:
    """
    Composite-time shadow engine operating on the 960x540 scene BEFORE foreground
    props are blitted, plus physically consistent prop-on-prop shadows:
      1. Separates tall standing structures (`cairn`, `ice_axe`, `theodolite`) from
         mounded (`rope`) and low-profile flat props (`gloves`, `notebook`) so
         ground-plane depth is never mistaken for vertical height.
      2. Applies a soft penumbra filter over directional cast shadows (`sun_dx = -0.52`,
         `sun_dy = +0.11` from the right horizon glow) while preserving tight
         high-contrast contact AO underneath object bases.
      3. Projects parallax prop-on-prop cast shadows (`ice_axe` -> `cairn`, and
         `theodolite` left/center/right tripod legs -> `rope`, `gloves`, `notebook`)
         while keeping standalone `sprites/*.png` 100% shadow-free.
    """
    lip_front_2x: List[Point] = [(x * 2, y * 2) for (x, y) in [
        (0, 155), (14, 162), (36, 161), (54, 171), (82, 171),
        (100, 179), (124, 181), (156, 170), (170, 161),
        (194, 154), (225, 145), (236, 138),
    ]]
    top_left_2x: List[Point] = [(x * 2, y * 2) for (x, y) in CLIFF_TOP_LEFT_POLY]

    plat_top_y = [999] * 480
    plat_bot_y = [-1] * 480
    for x in range(474):
        plat_top_y[x] = int(round(interp_y(top_left_2x, x)))
        plat_bot_y[x] = int(round(interp_y(lip_front_2x, x)))

    cast_map: List[List[float]] = [[0.0 for _ in range(480)] for _ in range(380)]
    ao_map: List[List[float]] = [[0.0 for _ in range(480)] for _ in range(380)]

    def add_cast(sx: int, sy: int, strength: float) -> None:
        if 0 <= sx < 474 and 204 <= sy < 370:
            if plat_top_y[sx] <= sy <= plat_bot_y[sx]:
                if strength > cast_map[sy][sx]:
                    cast_map[sy][sx] = strength

    def add_ao(sx: int, sy: int, strength: float) -> None:
        if 0 <= sx < 474 and 204 <= sy < 370:
            if plat_top_y[sx] <= sy <= plat_bot_y[sx]:
                if strength > ao_map[sy][sx]:
                    ao_map[sy][sx] = strength

    sun_dx = -0.62
    sun_dy = 0.06

    # 1A. Tall standing cairn shadow & deep grounded rock-base contact AO
    cairn_cast_scale = 1.28
    for x in range(36, 282):
        ys = [y for y in range(28, 290) if s_cairn.get_px(x, y)[3] > 0]
        if not ys:
            continue
        y_base = max(ys)
        for oy in (-1, 0, 1, 2, 3, 4, 5):
            for ox in (-4, -3, -2, -1, 0, 1, 2):
                ao_str = 1.22 if oy <= 2 else (0.98 if oy == 3 else (0.78 if oy == 4 else 0.56))
                add_ao(x + ox, y_base + oy, ao_str)
        for py in ys:
            h = max(0, y_base - py)
            h_cast = h * cairn_cast_scale
            sx = int(round(x + h_cast * sun_dx))
            sy = int(round(y_base + h_cast * sun_dy))
            s_val = 1.25 * max(0.52, 1.0 - h / 460.0)
            for ox in (0, 1, 2, 3):
                w = 1.0 if ox <= 1 else (0.92 if ox == 2 else 0.78)
                add_cast(sx - ox, sy, s_val * w)
            if h > 16:
                for ox in (0, 1, 2):
                    add_cast(sx - ox, sy + 1, s_val * (0.92 if ox <= 1 else 0.78))
            if h > 70:
                add_cast(sx, sy + 2, s_val * 0.78)
                add_cast(sx - 1, sy + 2, s_val * 0.72)

    # 1B. Mounded coiled rope: smooth ground curve + capped real coil height (<= 22px)
    rope_base_curve: List[Point] = [
        (126, 286), (144, 298), (175, 298), (195, 288), (225, 282), (266, 280)
    ]
    for y in range(225, 304):
        for x in range(124, 270):
            if s_rope.get_px(x, y)[3] == 0:
                continue
            yb = interp_y(rope_base_curve, x)
            # Contact AO right along the lower touching contour
            if y >= yb - 4:
                for oy in (0, 1, 2):
                    for ox in (-2, -1, 0, 1):
                        add_ao(x + ox, y + oy, 0.72 if oy <= 1 else 0.45)
            h_coil = min(22.0, max(2.0, yb - y + 4.0))
            steps_c = max(2, int(round(h_coil * 0.6)))
            for st in range(1, steps_c + 1):
                dist = (st / float(steps_c)) * h_coil
                sx = int(round(x + dist * sun_dx * 0.85))
                sy = int(round(max(y, yb - 2.0) + dist * sun_dy))
                fade = 0.56 * (1.0 - 0.35 * (st / float(steps_c)))
                add_cast(sx, sy, fade)

    # 1C. Low-profile flat props & right brink cobbles: true elevation cast + crisp contact AO
    low_prop_specs = [
        (s_gloves, 5.5, 0.68, 0.50, (180, 380), (268, 330)),
        (s_notebook, 4.8, 0.72, 0.54, (180, 380), (268, 330)),
    ]
    if s_cobbles is not None:
        low_prop_specs.append((s_cobbles, 8.0, 0.76, 0.56, (376, 448), (236, 278)))
    if s_platform_cobbles is not None:
        low_prop_specs.append((s_platform_cobbles, 4.0, 0.56, 0.36, (38, 382), (246, 355)))
    for sprite, elev, ao_str, cast_str, (x_lo, x_hi), (y_lo, y_hi) in low_prop_specs:
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if sprite.get_px(x, y)[3] == 0:
                    continue
                # Tight 1-2px contact occlusion right beneath the object perimeter
                is_bottom_edge = any(sprite.get_px(x + ox, y + oy)[3] == 0 for ox, oy in ((-1, 0), (0, 1), (-1, 1)))
                if is_bottom_edge:
                    for oy in (0, 1, 2):
                        for ox in (-2, -1, 0):
                            add_ao(x + ox, y + oy, ao_str if oy <= 1 else ao_str * 0.58)
                # Subtle low-right directional cast shadow proportional to actual height
                for step in range(1, 7):
                    u = step / 6.0
                    sx = int(round(x + elev * u * sun_dx * 2.2))
                    sy = int(round(y + elev * u * sun_dy * 2.2))
                    add_cast(sx, sy, cast_str * (1.0 - 0.55 * u))

    # 2. Standing Ice Axe & Tripod Legs: rays along (sun_dx*h, sun_dy*h) with widening penumbra
    def _tpt(px: float, py: float) -> Tuple[int, int]:
        return (int(round(299.0 + (px - 295.0) * 0.875)), int(round(285.0 + (py - 290.0) * 0.875)))

    def _ipt(px: float, py: float) -> Tuple[int, int]:
        return (int(round(102.0 + (px - 102.0) * 0.86)), int(round(294.0 + (py - 294.0) * 0.86)))

    leg_rays = [
        (_tpt(240, 282), _tpt(285, 116), 0.52),  # Left tripod leg
        (_tpt(289, 311), _tpt(295, 116), 0.58),  # Center tripod leg
        (_tpt(356, 278), _tpt(306, 116), 0.52),  # Right tripod leg
        (_ipt(96, 310),  _ipt(143, 146), 0.60),  # Ice axe shaft
    ]
    for (bx, by), (tx, ty), strg in leg_rays:
        for oy in (-1, 0, 1, 2):
            for ox in (-2, -1, 0, 1, 2):
                dist_pad = math.hypot(ox, oy)
                if dist_pad <= 2.4:
                    add_ao(bx + ox, by + oy, 0.78 * (1.0 - dist_pad * 0.25))
        steps = 120
        for s in range(steps + 1):
            u = s / float(steps)
            px = bx + u * (tx - bx)
            py = by + u * (ty - by)
            h = max(0.0, by - py)
            sx = int(round(px + h * sun_dx))
            sy = int(round(by + h * sun_dy))
            fade = strg * max(0.24, 1.0 - u * 0.70)
            rad = 1 if u < 0.45 else 2
            for oy in range(0, rad + 1):
                for ox in range(-rad, 1):
                    add_cast(sx + ox, sy + oy, fade * (1.0 if (ox == 0 and oy == 0) else 0.75))

    # Smooth cast shadow penumbra with a gentle 3x3 tent filter (keeping tight contact AO unblurred)
    soft_cast = [[0.0 for _ in range(480)] for _ in range(380)]
    for y in range(205, 369):
        for x in range(1, 473):
            acc = (
                cast_map[y][x] * 4.0
                + (cast_map[y][x - 1] + cast_map[y][x + 1] + cast_map[y - 1][x] + cast_map[y + 1][x]) * 2.0
                + (cast_map[y - 1][x - 1] + cast_map[y - 1][x + 1] + cast_map[y + 1][x - 1] + cast_map[y + 1][x + 1])
            ) / 16.0
            soft_cast[y][x] = acc

    # Composite twilight shadow onto platform (`scene_960`) without harsh checkerboard speckles
    shad_target: RGB = (14, 15, 26)
    for y in range(204, 370):
        for x in range(474):
            s_val = max(ao_map[y][x], soft_cast[y][x])
            if s_val <= 0.04:
                continue
            r, g, b, a = scene_960.get_px(x, y)
            if a == 0:
                continue
            thr = ((BAYER_4X4[y & 3][x & 3] + 0.5) / 16.0 - 0.5) * 0.08
            darken = max(0.0, min(0.95, s_val * 0.80 + thr))
            nr = int(round(r * (1.0 - darken) + shad_target[0] * darken))
            ng = int(round(g * (1.0 - darken) + shad_target[1] * darken))
            nb = int(round(b * (1.0 - darken) + shad_target[2] * darken))
            scene_960.set_px(x, y, (nr, ng, nb, 255))

    # 3. Parallax Prop-on-Prop Shadows (unified right-side sunlight across Cairn, Rope, Gloves, Notebook)
    cairn_lit = s_cairn.clone()
    rope_lit = s_rope.clone()
    gloves_lit = s_gloves.clone()
    notebook_lit = s_notebook.clone()

    def _darken_px(canvas: Canvas, px: int, py: int, ds: float, target: RGB = (16, 15, 28)) -> None:
        if not (0 <= px < FG_W and 0 <= py < FG_H):
            return
        cr, cg, cb, ca = canvas.get_px(px, py)
        if ca == 0:
            return
        nr = int(round(cr * (1.0 - ds) + target[0] * ds))
        ng = int(round(cg * (1.0 - ds) + target[1] * ds))
        nb = int(round(cb * (1.0 - ds) + target[2] * ds))
        canvas.set_px(px, py, (nr, ng, nb, 255))

    # 3A. Ice Axe shadow onto Cairn with realistic depth-gap parallax (tight at head y~165, softer lower down y~260)
    cairn_shad_mask: List[List[float]] = [[0.0 for _ in range(290)] for _ in range(300)]
    for y in range(154, 290):
        u_ax = max(0.0, min(1.0, (y - 160.0) / 130.0))
        dx_gap = int(round(-4.0 - 5.0 * u_ax))
        dy_gap = int(round(1.0 + 1.0 * u_ax))
        str_ax = 0.58 - 0.18 * u_ax
        for x in range(85, 185):
            if s_ice_axe.get_px(x, y)[3] > 0:
                for ox, oy, mul in ((0, 0, 1.0), (-1, 0, 0.80), (0, 1, 0.72)):
                    cx, cy = x + dx_gap + ox, y + dy_gap + oy
                    if 0 <= cx < 290 and 0 <= cy < 300:
                        val = str_ax * mul
                        if val > cairn_shad_mask[cy][cx]:
                            cairn_shad_mask[cy][cx] = val
    for cy in range(154, 295):
        for cx in range(40, 285):
            if cairn_shad_mask[cy][cx] > 0.0:
                _darken_px(cairn_lit, cx, cy, cairn_shad_mask[cy][cx])

    # 3B. Left tripod leg shadow onto coiled rope (`rope_lit`)
    for y in range(225, 286):
        for x in range(235, 272):
            if s_theodolite.get_px(x, y)[3] > 0:
                for dx, dy, ds in ((-5, 1, 0.48), (-6, 1, 0.36), (-4, 1, 0.34)):
                    _darken_px(rope_lit, x + dx, y + dy, ds, (14, 14, 26))

    # 3C. Center tripod leg shadow across front leather glove tips (`gloves_lit`, surface ~5px above ground)
    cbx, cby = _tpt(289, 311)
    ctx, cty = _tpt(295, 116)
    for s in range(120):
        u = s / 119.0
        px = cbx + u * (ctx - cbx)
        py = cby + u * (cty - cby)
        h_above_gloves = max(0.0, (cby - 5.0) - py)
        gx = int(round(px + h_above_gloves * sun_dx))
        gy = int(round((cby - 3.0) + h_above_gloves * sun_dy))
        for oy in (0, 1, 2):
            for ox in (-1, 0, 1):
                _darken_px(gloves_lit, gx + ox, gy + oy, 0.42 if (ox == 0 and oy <= 1) else 0.25, (14, 14, 24))

    # 3D. Right tripod leg shadow across back-left paper corner of notebook (`notebook_lit`, surface ~4.5px above ground)
    rbx, rby = _tpt(356, 278)
    rtx, rty = _tpt(306, 116)
    for s in range(90):
        u = s / 89.0
        px = rbx + u * (rtx - rbx)
        py = rby + u * (rty - rby)
        h_above_nb = max(0.0, (rby - 4.0) - py)
        nx = int(round(px + h_above_nb * sun_dx))
        ny = int(round((rby + 2.0) + h_above_nb * sun_dy))
        for oy in (0, 1):
            for ox in (-1, 0):
                _darken_px(notebook_lit, nx + ox, ny + oy, 0.46 if (ox == 0 and oy == 0) else 0.28, (36, 30, 42))

    return cairn_lit, rope_lit, gloves_lit, notebook_lit


def build_fg_prop_masks_2x(
    grid_fg: List[List[RGB]],
    base_masks: Dict[str, List[List[bool]]],
) -> Dict[str, List[List[bool]]]:
    """
    Refines the foreground prop masks onto the 960x540 (2x) foreground grid directly
    using `grid_fg` colors and tight 2x bounding envelopes for crisp edges on every
    foreground object (cairn, rope, ice_axe, gloves, notebook).
    """
    fg_masks: Dict[str, List[List[bool]]] = {
        name: [[False for _ in range(FG_W)] for _ in range(FG_H)]
        for name in ("ice_axe", "notebook", "gloves", "rope", "cairn")
    }

    def in_base_dilated(name: str, fx: int, fy: int, rad: int = 1) -> bool:
        bx_c = fx // FG_SCALE
        by_c = fy // FG_SCALE
        bmask = base_masks[name]
        for oy in range(-rad, rad + 1):
            by = max(0, min(NATIVE_H - 1, by_c + oy))
            for ox in range(-rad, rad + 1):
                bx = max(0, min(NATIVE_W - 1, bx_c + ox))
                if bmask[by][bx]:
                    return True
        return False

    # 1. ICE AXE at 960x540 (2x)
    axe_head_2x: List[Point] = [(x * 2, y * 2) for (x, y) in [
        (56, 69), (68, 69), (75, 71), (84, 74), (92, 80),
        (90, 82), (82, 79), (75, 76), (68, 74), (56, 73),
    ]]
    for y in range(136, 314):
        for x in range(90, 186):
            in_head = point_in_poly(x, y, axe_head_2x)
            d_shaft = dist_point_to_seg(x, y, 143.0, 146.0, 96.0, 310.0)
            in_shaft = (y >= 144) and (d_shaft <= (5.2 if y < 280 else 4.0))
            if in_head or in_shaft:
                fg_masks["ice_axe"][y][x] = True

    # 2. NOTEBOOK & PEN at 960x540 (2x) — match exact clean isometric notebook silhouette
    nb_ref = render_clean_notebook_sprite_2x()
    for y in range(270, 312):
        for x in range(302, 378):
            if nb_ref.get_px(x, y)[3] > 0:
                fg_masks["notebook"][y][x] = True

    # 3. LEATHER GLOVES at 960x540 (2x) — match exact tightened weathered-gloves silhouette
    tail_2x: List[Point] = [(x * 2, y * 2) for (x, y) in [
        (79, 146), (89, 147), (95, 154), (105, 156), (105, 164), (96, 164), (90, 157), (79, 153),
    ]]
    gloves_ref = render_weathered_gloves_sprite_2x()
    for y in range(276, 328):
        for x in range(180, 274):
            if gloves_ref.get_px(x, y)[3] > 0:
                fg_masks["gloves"][y][x] = True

    # 4. COILED ROPE at 960x540 (2x) — solid, coherent rounded coil silhouette without ragged tail/fuzz
    coil_2x: List[Point] = [
        (129, 273), (132, 260), (141, 253), (158, 251), (176, 252),
        (189, 247), (204, 236), (218, 229), (231, 228), (242, 233),
        (251, 243), (256, 256), (255, 268), (248, 277), (235, 281),
        (216, 281), (195, 285), (182, 293), (162, 295), (144, 291),
        (133, 283),
    ]
    for y in range(226, 298):
        for x in range(126, 268):
            if fg_masks["gloves"][y][x] or fg_masks["ice_axe"][y][x]:
                continue
            if point_in_poly(x + 0.5, y + 0.5, coil_2x):
                fg_masks["rope"][y][x] = True
    despike_mask(fg_masks["rope"], min_neighbors=2)

    # 5. STACKED STONE CAIRN at 960x540 (2x)
    cairn_lower_2x: List[Point] = [(x * 2, y * 2) for (x, y) in [
        (36, 98), (102, 98), (114, 102), (137, 112), (138, 121),
        (125, 128), (96, 134), (65, 141), (38, 143), (20, 139),
        (20, 126), (27, 112),
    ]]
    for y in range(32, 290):
        for x in range(38, 280):
            if fg_masks["rope"][y][x] or fg_masks["gloves"][y][x]:
                continue
            col = grid_fg[y][x]
            r, g, b = col
            lum = rgb_lum(col)
            if y <= 206 and 68 <= x <= 216:
                if lum < 36.0 or r > b + 7:
                    fg_masks["cairn"][y][x] = True
            elif y > 196 and point_in_poly(x, y, cairn_lower_2x):
                if not is_platform_slate_or_shadow(col) or (68 <= x <= 204 and y <= 268):
                    fg_masks["cairn"][y][x] = True
    fill_scanline_spans(
        fg_masks["cairn"], 32, 284, 38, 278,
        exclude_masks=[fg_masks["rope"], fg_masks["gloves"]],
    )
    despike_mask(fg_masks["cairn"], min_neighbors=2)

    return fg_masks


def render_clean_rope_sprite_2x(
    grid_fg: List[List[RGB]],
    fg_masks: Dict[str, List[List[bool]]],
    theo_canvas: Canvas,
    palette: Sequence[RGB],
) -> Canvas:
    """
    Renders a coherent, crisp, shadow-free 960x540 (2x) braided climbing rope sprite:
      - Solid, hole-free rounded coil silhouette (no ragged bottom fuzz or flat cut edges).
      - Preserves authentic twisted dark-hemp & sunlit ivory-tan interior strands from
        `grid_fg`, bridging ONLY the narrow tripod-leg crossing along curved coil strands.
      - Replaces washed-out background slate along outer turns with crisp braided outer
        hemp strands and wraps the full sprite in a continuous 1px dark perimeter outline
        so the coil separates cleanly from the granite platform and cairn behind it.
    """
    c = Canvas(FG_W, FG_H)
    mask = fg_masks["rope"]

    # Build a clean solid body mask (1px inset from final outline, including area behind gloves edge)
    coil_solid_poly: List[Point] = [
        (130, 273), (133, 261), (142, 254), (158, 252), (176, 253),
        (189, 248), (204, 237), (218, 230), (231, 229), (242, 234),
        (250, 244), (255, 256), (254, 268), (247, 276), (235, 280),
        (216, 280), (195, 284), (182, 292), (162, 294), (144, 290),
        (134, 282),
    ]
    body = [[False for _ in range(FG_W)] for _ in range(FG_H)]
    for y in range(227, 297):
        for x in range(128, 267):
            if point_in_poly(x + 0.5, y + 0.5, coil_solid_poly):
                body[y][x] = True

    def is_theo_px(px: int, py: int) -> bool:
        return theo_canvas.get_px(px, py)[3] > 0

    def is_rope_leg_occ(px: int, py: int) -> bool:
        r, g, b = grid_fg[py][px]
        in_leg_strip = dist_point_to_seg(px, py, 254.0, 220.0, 238.5, 279.0) <= 5.4
        in_ferrule = (256 <= py <= 279 and 236 <= px <= 247 and abs(r - g) < 18 and r > 62)
        in_mahogany = (224 <= py <= 264 and 240 <= px <= 258 and r > g + 16 and r > b + 8 and r < 145)
        return is_theo_px(px, py) or in_leg_strip or in_ferrule or in_mahogany

    def is_bad_bg_bleed(px: int, py: int) -> bool:
        col = grid_fg[py][px]
        r, g, b = col
        lum = rgb_lum(col)
        return (
            is_platform_slate_or_shadow(col)
            or (b >= r + 4 and lum > 30.0)
            or (py >= 278 and px >= 186 and fg_masks["gloves"][py][px])
        )

    # Precompute 1..3px rim distance inside `body`
    rim_dist = [[9 for _ in range(FG_W)] for _ in range(FG_H)]
    for y in range(227, 297):
        for x in range(128, 267):
            if not body[y][x]:
                continue
            if any(not body[y + dy][x + dx] for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                rim_dist[y][x] = 1
    for d in (2, 3):
        for y in range(227, 297):
            for x in range(128, 267):
                if body[y][x] and rim_dist[y][x] == 9:
                    if any(rim_dist[y + dy][x + dx] == d - 1 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                        rim_dist[y][x] = d

    # Curated weathered climbing-rope hemp ramp for crisp outer turns & coherent edge definition
    rope_dark: RGB = (30, 22, 26)
    rope_shad: RGB = (58, 44, 42)
    rope_mid: RGB = (96, 76, 64)
    rope_warm: RGB = (146, 122, 98)
    rope_hi: RGB = (188, 166, 134)

    def _mix_rgb(a: RGB, b: RGB, t: float) -> RGB:
        tt = max(0.0, min(1.0, t))
        return (
            int(round(a[0] + (b[0] - a[0]) * tt)),
            int(round(a[1] + (b[1] - a[1]) * tt)),
            int(round(a[2] + (b[2] - a[2]) * tt)),
        )

    for y in range(227, 297):
        for x in range(128, 267):
            if not body[y][x]:
                continue

            # 1. Obtain authentic interior rope pixel (or curved-strand bridge across leg / bg bleed)
            if is_rope_leg_occ(x, y):
                cx_r, cy_r = 224.0, 258.0
                dx_c = (x - cx_r) / 1.32
                dy_c = y - cy_r
                rad = math.hypot(dx_c, dy_c)
                ang = math.atan2(dy_c, dx_c)
                cand_l: Optional[Tuple[float, RGB]] = None
                cand_r: Optional[Tuple[float, RGB]] = None
                for step in range(2, 26):
                    d_ang = step * 0.055
                    if cand_l is None:
                        lx = int(round(cx_r + math.cos(ang - d_ang) * rad * 1.32))
                        ly = int(round(cy_r + math.sin(ang - d_ang) * rad))
                        if 0 <= lx < FG_W and 0 <= ly < FG_H and body[ly][lx] and not is_rope_leg_occ(lx, ly) and not is_bad_bg_bleed(lx, ly):
                            cand_l = (d_ang, grid_fg[ly][lx])
                    if cand_r is None:
                        rx = int(round(cx_r + math.cos(ang + d_ang) * rad * 1.32))
                        ry = int(round(cy_r + math.sin(ang + d_ang) * rad))
                        if 0 <= rx < FG_W and 0 <= ry < FG_H and body[ry][rx] and not is_rope_leg_occ(rx, ry) and not is_bad_bg_bleed(rx, ry):
                            cand_r = (d_ang, grid_fg[ry][rx])
                    if cand_l is not None and cand_r is not None:
                        break
                if cand_l is not None and cand_r is not None:
                    t_side = cand_l[0] / (cand_l[0] + cand_r[0])
                    jit = (fbm_hash(x, y, 509) - 0.5) * 0.18
                    col = cand_r[1] if (t_side + jit) > 0.5 else cand_l[1]
                elif cand_l is not None:
                    col = cand_l[1]
                elif cand_r is not None:
                    col = cand_r[1]
                else:
                    col = rope_shad
            elif is_bad_bg_bleed(x, y):
                # Pull clean interior braided hemp color instead of cold cliff slate or glove edge
                col = rope_shad
                found = False
                for rad_s in (2, 4, 6, 8):
                    for ox, oy in ((0, -rad_s), (rad_s, 0), (-rad_s, 0), (0, rad_s), (rad_s, -rad_s), (-rad_s, -rad_s)):
                        nx, ny = x + ox, y + oy
                        if 0 <= nx < FG_W and 0 <= ny < FG_H and body[ny][nx] and not is_rope_leg_occ(nx, ny) and not is_bad_bg_bleed(nx, ny):
                            col = grid_fg[ny][nx]
                            found = True
                            break
                    if found:
                        break
            else:
                col = grid_fg[y][x]

            # 2. Give the outer 2px rim clear braided hemp strand definition so edges never fade out
            rd = rim_dist[y][x]
            if rd <= 2:
                # Approximate local outward normal from the clean body mask
                nx_o, ny_o = 0.0, 0.0
                for oy in (-2, -1, 0, 1, 2):
                    for ox in (-2, -1, 0, 1, 2):
                        if not body[y + oy][x + ox]:
                            nx_o += ox
                            ny_o += oy
                nlen = math.hypot(nx_o, ny_o)
                if nlen > 1e-4:
                    nx_o /= nlen
                    ny_o /= nlen
                # Directional morning sun from top-right (+x, -y)
                sun_rim = max(0.0, min(1.0, 0.45 + 0.42 * nx_o - 0.42 * ny_o))
                # Twisted hemp braid notch pattern running around the outer coil turn
                twist_wave = 0.5 + 0.5 * math.sin((x * 0.72 + y * 0.64) * 1.55)
                if rd == 1:
                    if twist_wave < 0.24:
                        rim_col = rope_dark
                    elif sun_rim > 0.58:
                        rim_col = _mix_rgb(rope_warm, rope_hi, (sun_rim - 0.58) / 0.42 * twist_wave)
                    elif sun_rim > 0.34:
                        rim_col = _mix_rgb(rope_mid, rope_warm, twist_wave * 0.85)
                    else:
                        rim_col = _mix_rgb(rope_shad, rope_mid, twist_wave * 0.70)
                    col = _mix_rgb(col, rim_col, 0.72)
                else:
                    # Subtle 2nd-pixel inner-rim bevel bridging cleanly into authentic interior strands
                    if rgb_lum(col) < 42.0 and sun_rim > 0.55 and twist_wave > 0.45:
                        col = _mix_rgb(col, rope_mid, 0.45)
                    elif col[2] > col[0] + 2:
                        col = _mix_rgb(col, rope_shad, 0.55)

            c.set_px(x, y, quantize_pixel_bayer(x, y, col, palette, dither_strength=0.18))

    # 3. Crisp continuous 1px dark exterior outline matching all other 960x540 sprites
    outline_col: RGBA = (18, 14, 22, 255)
    for y in range(226, 298):
        for x in range(127, 268):
            if body[y][x]:
                continue
            if any(body[y + dy][x + dx] for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                c.set_px(x, y, outline_col)

    return c


def render_fg_prop_sprite_2x(
    name: str,
    grid_fg: List[List[RGB]],
    fg_masks: Dict[str, List[List[bool]]],
    theo_canvas: Canvas,
    palette: Sequence[RGB],
) -> Canvas:
    """
    Renders a 960x540 foreground prop (`cairn`, `rope`, `ice_axe`, `gloves`, `notebook`)
    at 2x foreground pixel resolution, inpaints occluded pixels behind overlapping props
    (ice axe on cairn, theodolite left leg on cairn/rope), and snaps to the master palette.
    """
    c = Canvas(FG_W, FG_H)
    mask = fg_masks[name]
    dither = 0.22 if name in ("cairn", "rope", "gloves") else 0.10

    def is_theo_px(px: int, py: int) -> bool:
        return theo_canvas.get_px(px, py)[3] > 0

    def is_rope_leg_occ(px: int, py: int) -> bool:
        r, g, b = grid_fg[py][px]
        # Tight mask around ONLY the actual narrow tripod left-leg shaft & ferrule in banner.jpg
        in_leg_strip = dist_point_to_seg(px, py, 254.0, 220.0, 238.5, 279.0) <= 5.4
        in_ferrule = (256 <= py <= 279 and 236 <= px <= 247 and abs(r - g) < 18 and r > 62)
        in_mahogany = (224 <= py <= 264 and 240 <= px <= 258 and r > g + 16 and r > b + 8 and r < 145)
        return (
            is_theo_px(px, py)
            or in_leg_strip
            or in_ferrule
            or in_mahogany
            or (px >= 254 and is_platform_slate_or_shadow(grid_fg[py][px]))
        )

    for y in range(FG_H):
        for x in range(FG_W):
            if not mask[y][x]:
                continue

            if name == "cairn" and (fg_masks["ice_axe"][y][x] or is_theo_px(x, y)):
                rep_col: Optional[RGB] = None
                for dx in (10, -10, 14, -14, 18, -18, 22, -22):
                    nx = x + dx
                    if 0 <= nx < FG_W and mask[y][nx] and not fg_masks["ice_axe"][y][nx] and not is_theo_px(nx, y):
                        rep_col = grid_fg[y][nx]
                        break
                if rep_col is None:
                    rep_col = grid_fg[y][x]
                c.set_px(x, y, quantize_pixel_bayer(x, y, rep_col, palette, dither_strength=dither))
            elif name == "rope" and is_rope_leg_occ(x, y):
                # Bridge ONLY the narrow tripod-leg occlusion by sampling real unoccluded braided
                # rope pixels along the exact curved strand contour on both sides of the leg
                cx_r, cy_r = 224.0, 258.0
                dx_c = (x - cx_r) / 1.32
                dy_c = y - cy_r
                rad = math.hypot(dx_c, dy_c)
                ang = math.atan2(dy_c, dx_c)

                cand_l: Optional[Tuple[float, RGB]] = None
                cand_r: Optional[Tuple[float, RGB]] = None
                for step in range(2, 26):
                    d_ang = step * 0.055
                    if cand_l is None:
                        lx = int(round(cx_r + math.cos(ang - d_ang) * rad * 1.32))
                        ly = int(round(cy_r + math.sin(ang - d_ang) * rad))
                        if 0 <= lx < FG_W and 0 <= ly < FG_H and mask[ly][lx] and not is_rope_leg_occ(lx, ly):
                            cand_l = (d_ang, grid_fg[ly][lx])
                    if cand_r is None:
                        rx = int(round(cx_r + math.cos(ang + d_ang) * rad * 1.32))
                        ry = int(round(cy_r + math.sin(ang + d_ang) * rad))
                        if 0 <= rx < FG_W and 0 <= ry < FG_H and mask[ry][rx] and not is_rope_leg_occ(rx, ry):
                            cand_r = (d_ang, grid_fg[ry][rx])
                    if cand_l is not None and cand_r is not None:
                        break

                if cand_l is not None and cand_r is not None:
                    # Pick cleanly from the nearer genuine strand side (preserves crisp braided pixel contrast)
                    t_side = cand_l[0] / (cand_l[0] + cand_r[0])
                    jit = (fbm_hash(x, y, 509) - 0.5) * 0.18
                    rep_col = cand_r[1] if (t_side + jit) > 0.5 else cand_l[1]
                elif cand_l is not None:
                    rep_col = cand_l[1]
                elif cand_r is not None:
                    rep_col = cand_r[1]
                else:
                    # Fallback horizontal sample inside unoccluded coil
                    rep_col = (28, 22, 26)
                    for dx in (-12, 12, -16, 16, -20):
                        nx = x + dx
                        if 0 <= nx < FG_W and mask[y][nx] and not is_rope_leg_occ(nx, y):
                            rep_col = grid_fg[y][nx]
                            break

                # Keep outermost perimeter of right loop cleanly outlined
                if any(not mask[y + dy][x + dx] for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                    rep_col = (22, 18, 24)
                c.set_px(x, y, quantize_pixel_bayer(x, y, rep_col, palette, dither_strength=dither))
            else:
                col = grid_fg[y][x]
                c.set_px(x, y, quantize_pixel_bayer(x, y, col, palette, dither_strength=dither))

    return c



def render_prop_sprite_from_mask(
    name: str,
    grid: List[List[RGB]],
    masks: Dict[str, List[List[bool]]],
    palette: Sequence[RGB],
) -> Canvas:
    """
    Renders a 480x270 object layer (`alpine_hut`, etc.) conditioned directly on
    the reference `grid` and quantized to the 64-color master VGA palette.
    """
    c = Canvas(NATIVE_W, NATIVE_H)
    mask = masks[name]
    dither = 0.25 if name in ("cairn", "rope", "gloves") else 0.12

    for y in range(NATIVE_H):
        for x in range(NATIVE_W):
            if not mask[y][x]:
                continue
            col = grid[y][x]
            c.set_px(x, y, quantize_pixel_bayer(x, y, col, palette, dither_strength=dither))

    return c



# ==============================================================================
# PALETTE SWATCH CHART & MAIN PIPELINE ORCHESTRATOR
# ==============================================================================
def render_palette_swatch(palette: Sequence[RGB]) -> Canvas:
    cols = 8
    rows = (len(palette) + cols - 1) // cols
    cell = 24
    c = Canvas(cols * cell, rows * cell)
    for idx, col in enumerate(palette):
        r = idx // cols
        k = idx % cols
        x0, y0 = k * cell, r * cell
        for y in range(y0, y0 + cell):
            for x in range(x0, x0 + cell):
                if x == x0 or y == y0 or x == x0 + cell - 1 or y == y0 + cell - 1:
                    c.set_px(x, y, (12, 14, 24, 255))
                else:
                    c.set_px(x, y, (col[0], col[1], col[2], 255))
    return c



def build_all(ref_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sprites_dir = os.path.join(out_dir, "sprites")
    layers_dir = os.path.join(out_dir, "layers")
    os.makedirs(sprites_dir, exist_ok=True)
    os.makedirs(layers_dir, exist_ok=True)

    print(f"[1/5] Decoding reference image {ref_path} (480x270 terrain + {FG_W}x{FG_H} foreground)...")
    w, h, rs, raw = load_reference_jpeg(ref_path)
    grid = sample_demixeled_grid(w, h, rs, raw, NATIVE_W, NATIVE_H)
    grid_fg = sample_demixeled_grid(w, h, rs, raw, FG_W, FG_H)

    print("[2/5] Extracting 64-color master VGA palette & reconstructing unoccluded sky field...")
    palette = extract_master_palette(grid, n_colors=64)
    swatch = render_palette_swatch(palette)
    swatch.save_png(os.path.join(out_dir, "palette_swatch.png"))

    sky_field = build_sky_color_field(grid, w, h, rs, raw)
    masks = build_prop_masks(grid, sky_field)
    fg_masks = build_fg_prop_masks_2x(grid_fg, masks)

    print("[3/5] Rendering pixel-art terrain layers (480x270)...")
    l1_sky = render_sky_layer_from_field(grid, sky_field, palette)
    l2_bg_mtns = render_background_mountains_layer(grid, palette)
    l3_mid_ridge = render_midground_ridge_layer(grid, masks, palette)
    l3b_hut = render_alpine_hut_sprite_2x()
    l4_cliff = render_foreground_cliff_layer(grid, masks, palette)

    l1_sky.save_png(os.path.join(layers_dir, "01_background_sky.png"))
    l2_bg_mtns.save_png(os.path.join(layers_dir, "02_background_mountains.png"))
    l3_mid_ridge.save_png(os.path.join(layers_dir, "03_midground_ridge.png"))
    l3b_hut.save_png(os.path.join(layers_dir, "03b_alpine_hut.png"))
    l4_cliff.save_png(os.path.join(layers_dir, "04_foreground_cliff.png"))

    print(f"[4/5] Rendering {FG_W}x{FG_H} (2x) & high-density (4x) shadow-free foreground object assets...")
    s_theodolite = render_polished_theodolite_sprite_2x()
    s_cairn = render_faithful_cairn_sprite_2x(grid_fg, fg_masks, palette)
    s_rope = render_clean_rope_sprite_2x(grid_fg, fg_masks, s_theodolite, palette)
    s_ice_axe_2x = render_polished_ice_axe(scale_from_2x=1)
    s_ice_axe_4x = render_polished_ice_axe(scale_from_2x=2)
    s_gloves = render_weathered_gloves_sprite_2x(palette)
    s_notebook = render_clean_notebook_sprite_2x()
    s_cobbles = render_right_cobbles_sprite_2x()
    s_platform_cobbles = render_platform_cobbles_sprite_2x()

    s_cairn.save_png(os.path.join(layers_dir, "05_cairn.png"))
    s_rope.save_png(os.path.join(layers_dir, "06_rope.png"))
    s_ice_axe_4x.save_png(os.path.join(layers_dir, "07_ice_axe.png"))
    s_gloves.save_png(os.path.join(layers_dir, "08_gloves.png"))
    s_notebook.save_png(os.path.join(layers_dir, "09_notebook.png"))
    s_theodolite.save_png(os.path.join(layers_dir, "10_theodolite.png"))
    s_cobbles.save_png(os.path.join(layers_dir, "11_right_cobbles.png"))
    s_platform_cobbles.save_png(os.path.join(layers_dir, "12_platform_cobbles.png"))

    # Save cropped 1x + upscaled previews for all shadow-free sprites
    l3b_hut.crop_non_empty(pad=3).save_png(os.path.join(sprites_dir, "alpine_hut.png"))
    l3b_hut.crop_non_empty(pad=3).upscale_nearest(4).save_png(os.path.join(sprites_dir, "alpine_hut_4x.png"))

    fg_sprite_map = {
        "cairn": s_cairn,
        "rope": s_rope,
        "gloves": s_gloves,
        "notebook": s_notebook,
        "theodolite": s_theodolite,
        "right_cobbles": s_cobbles,
    }
    for name, layer_canvas in fg_sprite_map.items():
        cropped = layer_canvas.crop_non_empty(pad=4)
        cropped.save_png(os.path.join(sprites_dir, f"{name}.png"))
        cropped.upscale_nearest(2).save_png(os.path.join(sprites_dir, f"{name}_2x.png"))
        cropped.upscale_nearest(4).save_png(os.path.join(sprites_dir, f"{name}_4x.png"))

    # Save native 1920x1080-density ice axe sprite + 2x preview
    axe_cropped_4x = s_ice_axe_4x.crop_non_empty(pad=8)
    axe_cropped_4x.save_png(os.path.join(sprites_dir, "ice_axe.png"))
    axe_cropped_4x.upscale_nearest(2).save_png(os.path.join(sprites_dir, "ice_axe_2x.png"))

    print("[5/5] Computing composite-time shadows & compositing 960x540 / 1920x1080 banners...")
    terrain_base = Canvas(NATIVE_W, NATIVE_H)
    for layer in (l1_sky, l2_bg_mtns, l3_mid_ridge, l4_cliff):
        terrain_base.blit(layer)

    scene_960 = terrain_base.upscale_nearest(FG_SCALE)
    # Composite 2x alpine hut + clean westward foundation & slope shadow onto the 960x540 ridge bench
    hut_shad: Dict[Tuple[int, int], float] = {}
    for x in range(676, 707):
        ys_h = [y for y in range(385, 416) if l3b_hut.get_px(x, y)[3] > 0]
        if not ys_h:
            continue
        yb = max(ys_h)
        # Tight foundation contact AO beneath the hut base
        for oy in (0, 1, 2):
            for ox in (-2, -1, 0):
                key = (x + ox, yb + oy)
                val = 0.60 if oy <= 1 else 0.36
                if val > hut_shad.get(key, 0.0):
                    hut_shad[key] = val
        # Short westward cast shadow along the green bench matching the unified sun angle
        h_hut = min(14.0, float(yb - min(ys_h)))
        for st in range(1, 6):
            u = st / 5.0
            sx = int(round(x - h_hut * u * 0.45))
            sy = int(round(yb + h_hut * u * 0.05))
            key = (sx, sy)
            val = 0.46 * (1.0 - 0.50 * u)
            if val > hut_shad.get(key, 0.0):
                hut_shad[key] = val
    for (sx, sy), ds in hut_shad.items():
        p = scene_960.get_px(sx, sy)
        if p[3] > 0:
            nr = int(round(p[0] * (1.0 - ds) + 16 * ds))
            ng = int(round(p[1] * (1.0 - ds) + 18 * ds))
            nb = int(round(p[2] * (1.0 - ds) + 28 * ds))
            scene_960.set_px(sx, sy, (nr, ng, nb, 255))
    scene_960.blit(l3b_hut)

    # Shrink theodolite slightly in the composite (0.875x) and seat right tripod foot clear behind notebook
    theo_comp_scale = 0.875
    theo_piv_x, theo_piv_y = 295.0, 290.0
    theo_dst_x, theo_dst_y = 299.0, 285.0

    def _scale_theo_canvas(src: Canvas, out_w: int, out_h: int, total_scale: float) -> Canvas:
        out = Canvas(out_w, out_h)
        dst_px = theo_dst_x * (out_w / float(src.w))
        dst_py = theo_dst_y * (out_h / float(src.h))
        x0 = max(0, int(math.floor(dst_px + (225.0 - theo_piv_x) * total_scale)))
        x1 = min(out_w - 1, int(math.ceil(dst_px + (370.0 - theo_piv_x) * total_scale)))
        y0 = max(0, int(math.floor(dst_py + (24.0 - theo_piv_y) * total_scale)))
        y1 = min(out_h - 1, int(math.ceil(dst_py + (320.0 - theo_piv_y) * total_scale)))
        for y in range(y0, y1 + 1):
            sy = int(round(theo_piv_y + (y - dst_py) / total_scale))
            if not (0 <= sy < src.h):
                continue
            for x in range(x0, x1 + 1):
                sx = int(round(theo_piv_x + (x - dst_px) / total_scale))
                if 0 <= sx < src.w:
                    px = src.get_px(sx, sy)
                    if px[3] > 0:
                        out.set_px(x, y, px)
        return out

    s_theodolite_comp_2x = _scale_theo_canvas(s_theodolite, FG_W, FG_H, theo_comp_scale)
    s_theodolite_comp_4x = _scale_theo_canvas(s_theodolite, FG_W * 2, FG_H * 2, 2.0 * theo_comp_scale)

    cairn_lit, rope_lit, gloves_lit, notebook_lit = apply_composite_shadows_2x(
        scene_960, s_cairn, s_rope, s_ice_axe_2x, s_gloves, s_notebook, s_theodolite_comp_2x, s_cobbles, s_platform_cobbles
    )
    scene_960_hd_base = scene_960.clone()
    for fg_layer in (s_platform_cobbles, cairn_lit, rope_lit, gloves_lit, notebook_lit, s_cobbles):
        scene_960_hd_base.blit(fg_layer)

    # Composite 1920x1080 high-density ice axe & shrunken crisp theodolite on top of 2x-upscaled 960x540 scene
    scene_1920 = scene_960_hd_base.upscale_nearest(2)
    scene_1920.blit(s_ice_axe_4x)
    scene_1920.blit(s_theodolite_comp_4x)

    for fg_layer in (s_platform_cobbles, cairn_lit, rope_lit, s_ice_axe_2x, gloves_lit, notebook_lit, s_cobbles, s_theodolite_comp_2x):
        scene_960.blit(fg_layer)

    fg_native_path = os.path.join(out_dir, "banner_pixel_960x540.png")
    upscale_hd_path = os.path.join(out_dir, "banner_pixel_1920x1080.png")
    diffuse_hd_path = os.path.join(out_dir, "banner_pixel_1920x1080_diffuse.png")

    scene_960.save_png(fg_native_path)
    scene_1920.save_png(upscale_hd_path)

    scene_1920_diffuse = apply_vintage_photo_filter(scene_1920)
    scene_1920_diffuse.save_png(diffuse_hd_path)
    print(f"Done! Saved shadow-free sprites, composite-time shadows, and 960x540 / 1920x1080 (+ diffuse) banners to {out_dir}")


def apply_vintage_photo_filter(src: Canvas) -> Canvas:
    """
    Combined 35mm Kodachrome slide + subtle eastern dawn sunbeams pass at 1920x1080:
      1. Gently projects faint crepuscular morning sunbeams fanning from the eastern
         sunrise notch across the twilight sky, distant peaks, and valley mist (kept
         strictly behind the foreground cliff face).
      2. Applies a 35mm Kodachrome slide optical pass preserving pixel-art block
         structure: 24% micro-diffusion, subtle corner-only lateral chromatic fringing,
         quarter-res warm highlight halation, rich twilight-blue shadows & warm
         golden-rose highlights, fine analog emulsion grain, and corner vignette.
    """
    w, h = src.w, src.h
    stride_src = w * 4

    # 1. Pre-pass: subtle eastern dawn sunbeams behind the foreground cliff
    pre = src.clone()
    sbuf = pre.buf
    ox_sun, oy_sun = 1965.0, 395.0
    cliff_pts_4x = [(x * 4, y * 4) for (x, y) in CLIFF_RIGHT_WALL_POLY]
    cliff_top_4x = [(x * 4, y * 4) for (x, y) in CLIFF_TOP_LEFT_POLY]
    cliff_top_row = [int(round(interp_y(cliff_top_4x, x))) if x <= 944 else 9999 for x in range(w)]

    for y in range(h):
        row = y * stride_src
        cliff_rx = int(round(interp_x(cliff_pts_4x, y))) if y >= 552 else 944
        dy = y - oy_sun
        for x in range(w):
            if x <= cliff_rx and y >= cliff_top_row[x]:
                continue
            dx = x - ox_sun
            dist = math.hypot(dx, dy)
            if dist < 1.0 or dist > 1720.0:
                continue
            ang = math.atan2(dy, dx)
            rel_a = ang - math.pi if ang > 0 else ang + math.pi
            if rel_a < -0.40 or rel_a > 0.70:
                continue
            env_a = math.exp(-((rel_a - 0.16) / 0.33) ** 2)
            env_d = max(0.0, 1.0 - dist / 1720.0) ** 1.2
            shaft = (
                0.48
                + 0.30 * math.sin(rel_a * 26.0 + 0.6)
                + 0.14 * math.sin(rel_a * 57.0 - 1.1)
                + 0.08 * math.cos(rel_a * 11.0)
            )
            ray = env_a * env_d * (max(0.0, shaft) ** 1.75) * 0.22
            if ray <= 0.003:
                continue
            ic = row + x * 4
            r = sbuf[ic]
            g = sbuf[ic + 1]
            b = sbuf[ic + 2]
            nr = int(r + (255.0 - r) * ray * 1.10 + 0.5)
            ng = int(g + (232.0 - g) * ray * 0.88 + 0.5)
            nb = int(b + (186.0 - b) * ray * 0.62 + 0.5)
            sbuf[ic] = 255 if nr > 255 else nr
            sbuf[ic + 1] = 255 if ng > 255 else ng
            sbuf[ic + 2] = 255 if nb > 255 else nb

    out = Canvas(w, h)
    obuf = out.buf

    # 2. Quarter-resolution highlight bloom / halation buffer (480x270)
    bw, bh = w // 4, h // 4
    b_len = bw * bh
    br = [0.0] * b_len
    bg = [0.0] * b_len
    bb = [0.0] * b_len
    for by in range(bh):
        row_off = (by * 4 + 1) * stride_src
        b_row = by * bw
        for bx in range(bw):
            idx = row_off + (bx * 4 + 1) * 4
            r = float(sbuf[idx])
            g = float(sbuf[idx + 1])
            b = float(sbuf[idx + 2])
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            if lum > 124.0:
                wt = ((lum - 124.0) / 131.0) ** 1.12
                bidx = b_row + bx
                br[bidx] = r * wt * 1.10
                bg[bidx] = g * wt * 1.00
                bb[bidx] = b * wt * 0.82

    def _box_blur_1d_h(arr: List[float], dst: List[float], rad: int) -> None:
        win = 2 * rad + 1
        for by in range(bh):
            base = by * bw
            s = arr[base] * rad
            for k in range(rad + 1):
                s += arr[base + k]
            for bx in range(bw):
                dst[base + bx] = s / win
                add_x = min(bw - 1, bx + rad + 1)
                sub_x = max(0, bx - rad)
                s += arr[base + add_x] - arr[base + sub_x]

    def _box_blur_1d_v(arr: List[float], dst: List[float], rad: int) -> None:
        win = 2 * rad + 1
        for bx in range(bw):
            s = arr[bx] * rad
            for k in range(rad + 1):
                s += arr[k * bw + bx]
            for by in range(bh):
                dst[by * bw + bx] = s / win
                add_y = min(bh - 1, by + rad + 1)
                sub_y = max(0, by - rad)
                s += arr[add_y * bw + bx] - arr[sub_y * bw + bx]

    tmp = [0.0] * b_len
    for ch in (br, bg, bb):
        for _ in range(3):
            _box_blur_1d_h(ch, tmp, 4)
            _box_blur_1d_v(tmp, ch, 4)

    # 3. 35mm Kodachrome slide optical pass (corner CA, 24% micro-diffusion, halation, slide toning & grain)
    cx_h = w * 0.5
    cy_h = h * 0.5
    dx_n = [(x - cx_h) / cx_h for x in range(w)]
    dy_n = [(y - cy_h) / cy_h for y in range(h)]
    dx2 = [v * v for v in dx_n]
    dy2 = [v * v for v in dy_n]

    rng_state = 0x35AA1978
    for y in range(h):
        ym1 = y - 1 if y > 0 else 0
        yp1 = y + 1 if y < h - 1 else h - 1
        row_c = y * stride_src
        row_u = ym1 * stride_src
        row_d = yp1 * stride_src
        yv = dy2[y]

        by_f = (y - 1.5) * 0.25
        if by_f < 0.0:
            by_f = 0.0
        elif by_f > bh - 1.001:
            by_f = bh - 1.001
        by0 = int(by_f)
        fy = by_f - by0
        ify = 1.0 - fy
        b_r0 = by0 * bw
        b_r1 = (by0 + 1) * bw

        for x in range(w):
            r2 = dx2[x] + yv
            ca_shift = int(round(dx_n[x] * r2 * 1.15))
            xr = max(0, min(w - 1, x + ca_shift))
            xb = max(0, min(w - 1, x - ca_shift))
            xm1 = x - 1 if x > 0 else 0
            xp1 = x + 1 if x < w - 1 else w - 1

            ic = row_c + x * 4
            ir_ca = row_c + xr * 4
            ib_ca = row_c + xb * 4
            il = row_c + xm1 * 4
            ir = row_c + xp1 * 4
            iu = row_u + x * 4
            id_ = row_d + x * 4

            r = sbuf[ir_ca] * 0.76 + (sbuf[il] + sbuf[ir] + sbuf[iu] + sbuf[id_]) * 0.06
            g = sbuf[ic + 1] * 0.76 + (sbuf[il + 1] + sbuf[ir + 1] + sbuf[iu + 1] + sbuf[id_ + 1]) * 0.06
            b = sbuf[ib_ca + 2] * 0.76 + (sbuf[il + 2] + sbuf[ir + 2] + sbuf[iu + 2] + sbuf[id_ + 2]) * 0.06

            # Bilinear optical halation / highlight bloom lookup
            bx_f = (x - 1.5) * 0.25
            if bx_f < 0.0:
                bx_f = 0.0
            elif bx_f > bw - 1.001:
                bx_f = bw - 1.001
            bx0 = int(bx_f)
            fx = bx_f - bx0
            w00 = (1.0 - fx) * ify
            w10 = fx * ify
            w01 = (1.0 - fx) * fy
            w11 = fx * fy
            i00 = b_r0 + bx0
            i10 = i00 + 1
            i01 = b_r1 + bx0
            i11 = i01 + 1
            blr = br[i00] * w00 + br[i10] * w10 + br[i01] * w01 + br[i11] * w11
            blg = bg[i00] * w00 + bg[i10] * w10 + bg[i01] * w01 + bg[i11] * w11
            blb = bb[i00] * w00 + bb[i10] * w10 + bb[i01] * w01 + bb[i11] * w11

            # Soft screen blend of warm optical halation
            r = r + blr * (1.0 - r / 255.0) * 0.42
            g = g + blg * (1.0 - g / 255.0) * 0.38
            b = b + blb * (1.0 - b / 255.0) * 0.34

            # Kodachrome slide emulsion curve: rich twilight blues + warm golden-rose highlights
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            hi_t = max(0.0, (lum - 115.0) / 140.0)
            sh_t = max(0.0, (95.0 - lum) / 95.0)
            r = lum + (r - lum) * 1.08 + 7.0 * hi_t - 3.5 * sh_t + 3.5
            g = lum + (g - lum) * 1.05 + 2.5 * hi_t - 1.5 * sh_t + 2.5
            b = lum + (b - lum) * 1.09 - 5.5 * hi_t + 6.5 * sh_t + 5.0

            # Fine analog slide grain + corner vignette
            rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
            grain = (((rng_state >> 24) & 0xFF) - 127.5) * (4.0 / 127.5)
            vig = 1.0 - 0.19 * (r2 * 0.58)

            nr = int((r + grain) * vig + 0.5)
            ng = int((g + grain * 0.95) * vig + 0.5)
            nb = int((b + grain * 0.85) * vig + 0.5)

            obuf[ic] = 0 if nr < 0 else (255 if nr > 255 else nr)
            obuf[ic + 1] = 0 if ng < 0 else (255 if ng > 255 else ng)
            obuf[ic + 2] = 0 if nb < 0 else (255 if nb > 255 else nb)
            obuf[ic + 3] = 255

    return out


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reference_jpeg = os.path.join(root_dir, "assets", "banner.jpg")
    out_directory = os.path.join(root_dir, "assets", "pixel_art")
    build_all(reference_jpeg, out_directory)

# stroke_palette_pipeline3_clean_FINAL_v7_masks2_batch.py
# v7_masks2 + batch folder mode:
# - --input can be a file OR a folder
# - For folder: creates one subfolder per image under --out_dir, named by image stem
# - Outputs per image (inside its subfolder):
#     masked_core_dt.png
#     stroke_palette.png
#     binary_strokes.png
#     mask_XX_R_G_B.png
#     mask_XX_R_G_B_color.png
# - Terminal output format stays the same for each image (Top colors: ...)

import argparse
import os
import math
import csv
import time  # 🌟 新增這行：匯入時間模組
from collections import defaultdict
from typing import List
from pathlib import Path
import cv2
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageOps
# =========================
# PARAMETERS (MUST match your v7)
# =========================
# --- 1. 背景與前處理 ---
WHITE_DIST_THR = 70.0  # [背景過濾] 判斷背景(白紙)的RGB距離門檻。數值越大，越容易把淺色塗鴉誤認為白紙濾掉。

QUANT_STEP = 8  # [色彩量化] 初始顏色量化的步長。數字越大，一開始提取的顏色越少且越平滑，8 是適中的降噪值。

# --- 2. 彩色與灰階判定 ---
SAT_THR = 30       # [飽和度門檻] 判斷是否為「彩色」的飽和度(S)下限。低於 30 會被視為灰階/黑色處理。
HUE_BIN_DEG = 18   # [色相切塊] 初始將色相環(Hue)切塊的度數大小。18度代表將360度的色相環切成20等分。

# 針對「很亮但很淡」的顏色 (例如淡淡的淺藍色或粉紅色)，放寬飽和度限制，避免被當作灰色：
SAT_THR_LIGHT = 15 # 放寬後的飽和度下限。
V_LIGHT_MIN = 170  # 必須亮度(V)大於 170 才會套用上述放寬標準。

GRAY_BINS = 4      # [灰階切塊] 將非彩色的部分切成幾個明暗等級 (例如純黑、深灰、淺灰、白)。

# --- 3. 筆跡萃取與雜訊過濾 ---
CORE_DIST = 1         # 尋找筆畫「核心」的內縮像素距離，用來避開筆跡邊緣的漸層雜訊與反鋸齒邊緣。
MIN_STROKE_AREA = 30  # 連通區域的最小像素面積。小於 30 像素的微小筆畫一開始就會被徹底忽略。

# [全局去雜訊] 在顏色分析前，去除掃描器產生的細小斑點：
SPECKLE_AREA_MAX = 120   # 被判定為「掃描雜訊」的最大像素面積 (必須小於此值)。
SPECKLE_DIM_MAX  = 6     # 該雜訊的邊界框長或寬必須小於此值 (避免誤刪小朋友畫的長條形細線)。

# [各色去雜訊] 在顏色分離後，對「每個單獨的顏色遮罩」進行內部雜訊清理：
COLOR_SPECKLE_AREA_MAX = 160  # 各顏色遮罩內部殘留孤立小點的最大清除面積。
COLOR_SPECKLE_DIM_MAX  = 10   # 承上，孤立小點的長寬上限。

# --- 4. 斷裂筆觸縫補 (確保線條完整性) ---
# ---- NEW: mask connection/denoise for line completeness ----
# 手繪線條常會因為筆壓/掃描造成「斷裂成碎片」；此步驟只在每個顏色 mask 內做非常輕微的連接與去雜訊。
# 目標：線條更完整、碎片更少；不改變顏色分類本身。
MASK_CONNECT_ENABLE = True
MASK_CLOSE_KSIZE = 3          # 3=很保守；想更黏一點可改 5
MASK_CLOSE_ITERS = 1          # 1=很保守；想更黏一點可改 2
MASK_OPEN_KSIZE = 0           # 0=不做 opening；若小噪點很多可用 3
MASK_OPEN_ITERS = 1
# --- 5. 消除深色重疊邊緣 (Absorb outline variants) ---
# 小朋友畫畫時邊緣用力塗抹會產生「深色輪廓」，將其吸收回主顏色中：
OUTLINE_ABSORB_MAX_PROP = 0.12 # 被當作「深色輪廓」吸收的顏色，其佔比不能超過整張畫的 12%。
OUTLINE_ADJ_MIN_RATIO   = 0.55 # 這個深色輪廓必須有 55% 的面積緊貼著主顏色，才會被吸收。
OUTLINE_DE_MAX          = 55.0 # 被吸收的顏色與主顏色的 CIEDE2000 色差上限。
OUTLINE_RGB_DIST_MAX    = 95.0 # 被吸收的顏色與主顏色的 RGB 距離上限。
OUTLINE_DV_MIN          = 18.0 # 被吸收的顏色必須比主顏色「暗」至少 18 的亮度(V)。


# NEW: 防止「超小顏色」被當成新色輸出（會造成一堆 mask_*）
# 只要某個顏色佔比/像素數太小，我們就不把它當成獨立顏色，
# 而是把它的像素重新指派到「最接近的保留顏色」。
# 這樣雜點會被併回主色，而不是直接刪掉筆跡。
MIN_COLOR_PROP = 0.01      # 0.2% 以下視為太小顏色（可調）
MIN_COLOR_PIXELS = 600      # 像素數小於此也視為太小（可調）

# --- 7. 強制保護機制 (避免關鍵小特徵被吞噬) ---
# [強制保護黑色] (例如角落的簽名或編號)：
# NEW: 黑色小字(例如左下角的小 1 / 2)常常像素很少，
# 但我們希望它不要被當成雜訊或被比例門檻濾掉。
# 只要在筆跡區域出現足夠「很黑」的像素，就強制把黑色加入 palette。
FORCE_BLACK = True
BLACK_V_MAX = 70            # HSV 的 V <= 70 視為黑（可調）
BLACK_S_MAX = 80            # HSV 的 S <= 80（避免深色彩被當黑）
MIN_BLACK_PIXELS = 40       # 黑色像素數 >= 這個才強制加入（可調）

BLACK_TEXT_ROI_ENABLE = True # 是否只在圖片的特定區域(四個角落)強制尋找黑色文字。
BLACK_TEXT_ROI_X_FRAC = 0.25 # 角落區域佔圖片寬度的比例 (0.25 = 左右各25%)。
BLACK_TEXT_ROI_Y_FRAC = 0.25 # 角落區域佔圖片高度的比例 (0.25 = 底部25%)。

BLACK_COMP_MIN_AREA = 35        # 黑字筆畫的最小面積。
BLACK_COMP_MAX_BBOX_AREA = 8000 # 黑字邊界框的最大面積 (避免把巨大的黑影當作文字)。
BLACK_COMP_MAX_DIM = 140        # 黑字的最大長寬限制。


# NEW: 強制保留「紅色」(小面積也要保住，避免被黃/橘吸收)
FORCE_RED = True

# =========================
# RED MERGE TWEAK (MINIMAL CHANGE)
# =========================
# 目的：
# - 解決「同一支紅色筆跡，因掃描/壓力/亮度不同，被分成兩個紅色」的問題
# - 僅在『紅色家族』內放寬合併條件
# - 不影響：紅/橘/黃 之間的區分，也不影響粉紅/紫色
#
# 規則（很保守）：
# - 兩色都必須落在 red sector（hue_sector_179 == 0）
# - hue 差距 <= RED_MERGE_HUE_MAX
# - Lab ΔE <= RED_MERGE_DE_MAX
#
# 這只處理「明顯是同一個紅色」但被切成兩色的情況
# =========================

RED_MERGE_HUE_MAX = 34.0   # 只對紅色放寬（避免紅被切成兩色）
RED_MERGE_DE_MAX  = 48.0   # ΔE 上限，避免紅/粉紅誤併
RED_H_MAX = 8               # HSV Hue <= 8 視為紅（OpenCV H:0..179）
RED_H_MIN = 171             # 或 Hue >= 171 視為紅（wrap-around）
RED_S_MIN = 80              # 飽和度至少要高，才算紅
RED_V_MIN = 80              # 亮度至少要高，避免暗紅被當黑
MIN_RED_PIXELS = 60         # 紅色像素數 >= 這個才強制加入（可調）

# --- 8. 🌟 初始語意合併門檻 (決定顏色是否要分家的根源) ---
HUE_THR_DEG = 18   #一般顏色的色相容忍度 (原本 10 太嚴格，改為 18 度以內視為同一支筆)。
DE_THR_SAT = 40.0  # 初始顏色分群時，彩色與彩色的 deltaE 色差容忍度。
DE_THR_GRAY = 35.0 # 初始顏色分群時，灰階與灰階的 deltaE 色差容忍度。

# --- 9. 極端狀況處理 ---
SEMANTIC_DOM = 0.95  # 當某單一顏色佔整張圖 95% 以上時，啟動極端合併模式 (通常發生在大面積塗黑的畫)。
SEMANTIC_DIST = 70.0 # 極端合併模式下的 RGB 距離容忍度。

# --- 10. 輸出設定 ---
TOP_K = 20  # 最終輸出的顏色種類上限 (最多只保留 20 種顏色，避免報表過度肥大)。OUT_CORE = "masked_core_dt.png"
OUT_CORE = "masked_core_dt.png"
OUT_PALETTE = "stroke_palette.png"

OUT_BINARY = "binary_strokes.png"
MASK_BIN_FMT = "mask_{idx:02d}_{r}_{g}_{b}.png"
MASK_COLOR_FMT = "mask_{idx:02d}_{r}_{g}_{b}_color.png"
# =========================


def rgb_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def hue_diff_deg(h1_179, h2_179):
    a = float(h1_179) * 2.0
    b = float(h2_179) * 2.0
    d = abs(a - b)
    return min(d, 360.0 - d)


# --- NEW: adaptive hue merge thresholds to avoid merging distinct standard colors ---
def _is_red_band(h_179: float) -> bool:
    # OpenCV H in [0,179]; treat near 0/179 as red family (wrap-around)
    return (h_179 <= 8) or (h_179 >= 171)

def _is_yellow_orange_band(h_179: float) -> bool:
    # roughly orange->yellow in OpenCV hue space
    return 10 <= h_179 <= 40

def adaptive_hue_merge_thr(h1_179: float, h2_179: float) -> float:
    # keep reds merge-friendly (fix red split), but keep orange/yellow strict (avoid orange->yellow or red->orange merging)
    if _is_red_band(h1_179) and _is_red_band(h2_179):
        return 30.0
    if _is_yellow_orange_band(h1_179) and _is_yellow_orange_band(h2_179):
        return 8.0
    # default: keep original strictness
    return float(HUE_THR_DEG)


# --- merge guards for warm colors (prevent red/orange/yellow collapse in real drawings) ---
SAT_STRICT = 50  # stricter saturation threshold to apply sector guard during merging
V_STRICT_MIN = 60

def hue_sector_179(h_179: float) -> int:
    """Coarse hue sector in OpenCV-HSV hue space [0,179].
    0=red,1=orange,2=yellow,3=green,4=cyan,5=blue,6=purple"""
    h = float(h_179) % 180.0
    if h < 10 or h >= 170:
        return 0  # red (wrap-around)
    if h < 22:
        return 1  # orange
    if h < 35:
        return 2  # yellow
    if h < 85:
        return 3  # green
    if h < 100:
        return 4  # cyan
    if h < 135:
        return 5  # blue
    return 6  # purple

def _is_chromatic_hsv(hsv_vec) -> bool:
    s = float(hsv_vec[1])
    v = float(hsv_vec[2])
    return (s >= SAT_THR) or ((s >= SAT_THR_LIGHT) and (v >= V_LIGHT_MIN))


def delta_e76(lab1, lab2):
    d = lab1 - lab2
    return float(np.sqrt((d * d).sum()))



# =========================
# NEW: mask merge by CIEDE2000 + size rules (v8_15 requirement)
# - Use REAL pixel RGB inside each existing mask to compute mean Lab.
# - Merge ONLY when ALL conditions are satisfied:
#     1) deltaE00 < 19
#     2) big mask proportion >= 30%
#     3) small mask proportion <= 3%
# - Merge means: OR pixels into big mask (NOT deleting strokes), then remove small mask entry.
# - After merging: palette / CSV / palette bar / filenames remain consistent with the final masks.
# =========================

def _lab_opencv_to_cie(lab_cv: np.ndarray) -> tuple:
    """Convert OpenCV Lab (L:0..255, a:0..255, b:0..255 with 128 offset) to CIE Lab (L*:0..100, a*:-128..127, b*:-128..127)."""
    L = float(lab_cv[0]) * (100.0 / 255.0)
    a = float(lab_cv[1]) - 128.0
    b = float(lab_cv[2]) - 128.0
    return (L, a, b)

def delta_e_ciede2000(lab1_cv: np.ndarray, lab2_cv: np.ndarray) -> float:
    """CIEDE2000 deltaE between two OpenCV-Lab vectors (shape (3,))."""
    # Implementation based on Sharma et al. (2005) reference formula.
    L1, a1, b1 = _lab_opencv_to_cie(lab1_cv)
    L2, a2, b2 = _lab_opencv_to_cie(lab2_cv)

    kL = 1.0
    kC = 1.0
    kH = 1.0

    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + 25.0**7))) if C_bar > 0 else 0.0

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.sqrt(a1p*a1p + b1*b1)
    C2p = math.sqrt(a2p*a2p + b2*b2)
    C_bar_p = 0.5 * (C1p + C2p)

    def _hp(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180.0:
            dh -= 360.0
        elif dh < -180.0:
            dh += 360.0
        dhp = dh
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    L_bar_p = 0.5 * (L1 + L2)

    if C1p * C2p == 0:
        h_bar_p = h1p + h2p
    else:
        hsum = h1p + h2p
        hdiff = abs(h1p - h2p)
        if hdiff > 180.0:
            h_bar_p = (hsum + 360.0) / 2.0 if hsum < 360.0 else (hsum - 360.0) / 2.0
        else:
            h_bar_p = hsum / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    delta_theta = 30.0 * math.exp(-((h_bar_p - 275.0) / 25.0) ** 2)
    R_C = 2.0 * math.sqrt((C_bar_p**7) / (C_bar_p**7 + 25.0**7)) if C_bar_p > 0 else 0.0
    S_L = 1.0 + (0.015 * ((L_bar_p - 50.0) ** 2)) / math.sqrt(20.0 + ((L_bar_p - 50.0) ** 2))
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -math.sin(math.radians(2.0 * delta_theta)) * R_C

    dE = math.sqrt(
        (dLp / (kL * S_L)) ** 2
        + (dCp / (kC * S_C)) ** 2
        + (dHp / (kH * S_H)) ** 2
        + R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
    )
    return float(dE)

def _mask_mean_lab_cv(arr_rgb: np.ndarray, mask255: np.ndarray) -> np.ndarray:
    """Compute mean Lab (OpenCV encoding) from REAL pixel RGB inside the mask."""
    if mask255 is None:
        return None
    m = (mask255 > 0)
    if not m.any():
        return None
    px = arr_rgb[m].reshape(-1, 3).astype(np.float32)
    mean_rgb = np.clip(px.mean(axis=0), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(mean_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    return lab

def merge_masks_by_ciede2000(arr_rgb: np.ndarray,
                            palette: list,
                            masks_by_color: dict,
                            de_thr: float = 19.0,
                            big_prop_thr: float = 0.15,
                            small_prop_thr: float = 0.03) -> tuple:
    """Merge existing masks based on size + CIEDE2000 computed from mask-real pixels.
    Returns (new_palette, new_masks_by_color).
    """
    if not palette:
        return palette, masks_by_color

    rgbs = [rgb for rgb, _ in palette if rgb in masks_by_color]
    if not rgbs:
        return palette, masks_by_color

    counts = {rgb: int((masks_by_color[rgb] > 0).sum()) for rgb in rgbs}
    total = int(sum(counts.values())) or 1
    props = {rgb: counts[rgb] / float(total) for rgb in rgbs}

    bigs = [rgb for rgb in rgbs if (props.get(rgb, 0.0) >= float(big_prop_thr)) and (counts.get(rgb, 0) > 0)]
    smalls = [rgb for rgb in rgbs if (props.get(rgb, 0.0) <= float(small_prop_thr)) and (counts.get(rgb, 0) > 0)]

    if not bigs or not smalls:
        new_palette = [(rgb, props[rgb]) for rgb in rgbs]
        new_palette.sort(key=lambda x: x[1], reverse=True)
        return new_palette, masks_by_color

    lab_cache = {}
    for rgb in set(bigs + smalls):
        lab = _mask_mean_lab_cv(arr_rgb, masks_by_color.get(rgb))
        if lab is not None:
            lab_cache[rgb] = lab

    merged_into = {}  # small -> big

    for s_rgb in smalls:
        if s_rgb not in lab_cache:
            continue
        best_big = None
        best_de = 1e9
        for b_rgb in bigs:
            if b_rgb == s_rgb:
                continue
            if b_rgb not in lab_cache:
                continue
            de = delta_e_ciede2000(lab_cache[s_rgb], lab_cache[b_rgb])
            if de < float(de_thr) and de < best_de:
                best_de = de
                best_big = b_rgb
        if best_big is not None:
            merged_into[s_rgb] = best_big

    if not merged_into:
        new_palette = [(rgb, props[rgb]) for rgb in rgbs]
        new_palette.sort(key=lambda x: x[1], reverse=True)
        return new_palette, masks_by_color

    for s_rgb, b_rgb in merged_into.items():
        if s_rgb not in masks_by_color or b_rgb not in masks_by_color:
            continue
        masks_by_color[b_rgb] = cv2.bitwise_or(masks_by_color[b_rgb], masks_by_color[s_rgb])
        masks_by_color.pop(s_rgb, None)

    final_rgbs = [rgb for rgb in rgbs if rgb in masks_by_color]
    final_counts = {rgb: int((masks_by_color[rgb] > 0).sum()) for rgb in final_rgbs}
    final_total = int(sum(final_counts.values())) or 1

    new_palette = [(rgb, final_counts[rgb] / float(final_total)) for rgb in final_rgbs if final_counts[rgb] > 0]
    new_palette.sort(key=lambda x: x[1], reverse=True)

    new_palette = new_palette[:TOP_K]
    s = sum(p for _, p in new_palette) or 1.0
    new_palette = [(rgb, p / s) for rgb, p in new_palette]

    return new_palette, masks_by_color


def is_background(arr_rgb):
    diff = 255.0 - arr_rgb.astype(np.float32)
    dist = np.sqrt((diff * diff).sum(axis=2))
    return dist <= float(WHITE_DIST_THR)


def estimate_paper_rgb(arr_rgb: np.ndarray) -> tuple:
    """Estimate paper/background color from image borders.
    Works for photos/scans where paper is not pure white (shadows/temperature).
    Returns an (R,G,B) tuple.
    """
    h, w = arr_rgb.shape[:2]
    bw = int(max(8, round(min(h, w) * 0.05)))
    # collect border pixels
    top = arr_rgb[:bw, :, :].reshape(-1, 3)
    bot = arr_rgb[h - bw :, :, :].reshape(-1, 3)
    left = arr_rgb[:, :bw, :].reshape(-1, 3)
    right = arr_rgb[:, w - bw :, :].reshape(-1, 3)
    samp = np.concatenate([top, bot, left, right], axis=0)
    if samp.size == 0:
        return (255, 255, 255)
    hsv = cv2.cvtColor(samp.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    # pick bright & low-saturation candidates (likely paper)
    cand = samp[(hsv[:, 2] >= 180) & (hsv[:, 1] <= 45)]
    if cand.size == 0:
        # fallback to brightest 10%
        v = hsv[:, 2]
        thr = np.percentile(v, 90)
        cand = samp[v >= thr]
    if cand.size == 0:
        return (255, 255, 255)
    med = np.median(cand.astype(np.float32), axis=0)
    return (int(round(med[0])), int(round(med[1])), int(round(med[2])))


def is_background_auto(arr_rgb: np.ndarray) -> np.ndarray:
    """Background mask based on estimated paper color (more robust than pure white dist)."""
    paper = estimate_paper_rgb(arr_rgb)
    paper_arr = np.array(paper, dtype=np.float32).reshape(1, 1, 3)
    diff = arr_rgb.astype(np.float32) - paper_arr
    dist = np.sqrt((diff * diff).sum(axis=2))

    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    # allow slightly tinted paper as background when low-sat and close to estimated paper color
    # (helps with uneven lighting, and with "thin paint" areas that expose paper texture)
    bg = (
        (dist <= float(WHITE_DIST_THR))
        | ((hsv[..., 1] <= 35) & (hsv[..., 2] >= 160) & (dist <= float(WHITE_DIST_THR) + 20.0))
        | ((hsv[..., 1] <= 60) & (hsv[..., 2] >= 120) & (dist <= float(WHITE_DIST_THR) + 35.0))
    )
    return bg


def quantize_rgb(pixels_rgb, step):
    if step <= 1:
        return pixels_rgb
    return (pixels_rgb // step) * step


def dominant_color_mode(pixels_rgb, quant_step=16):
    if pixels_rgb is None or pixels_rgb.size == 0:
        return (0, 0, 0)
    q = quantize_rgb(pixels_rgb.astype(np.uint8), quant_step)
    packed = (q[:, 0].astype(np.int32) << 16) | (q[:, 1].astype(np.int32) << 8) | q[:, 2].astype(np.int32)
    vals, counts = np.unique(packed, return_counts=True)
    best = int(vals[np.argmax(counts)])
    return (int((best >> 16) & 255), int((best >> 8) & 255), int(best & 255))


def draw_palette_bar(colors_with_prop, out_path, w=900, h=120):
    if not colors_with_prop:
        return
    colors_with_prop = sorted(colors_with_prop, key=lambda x: x[1], reverse=True)
    bar = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(bar)

    x = 0
    for rgb, p in colors_with_prop:
        if x >= w:            # 🌟 新增防呆：如果 x 已經超過或等於圖片寬度，就立刻停止
            break
        bw = int(round(p * w))
        if bw <= 0:
            continue
        draw.rectangle([x, 0, min(w, x + bw), h], fill=rgb)
        x += bw

    if x < w:
        draw.rectangle([x, 0, w, h], fill=colors_with_prop[-1][0])

    bar.save(out_path)


def semantic_merge_with_mapping(color_accum):
    items = sorted(color_accum.items(), key=lambda x: x[1], reverse=True)
    merged = []  # {"rgb": rgb, "cnt": cnt, "lab": lab, "hsv": hsv, "members":[orig...]}

    for rgb, cnt in items:
        rgb_arr = np.array([[rgb]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
        hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV).reshape(3).astype(np.float32)

        placed = False
        for m in merged:
            de = delta_e76(lab, m["lab"])

            a_ch = _is_chromatic_hsv(hsv)
            b_ch = _is_chromatic_hsv(m["hsv"])

            # Case 1) both chromatic -> use hue sector + hue + deltaE
            if a_ch and b_ch:
                dh = hue_diff_deg(hsv[0], m["hsv"][0])
                sec_a = hue_sector_179(hsv[0])
                sec_b = hue_sector_179(m["hsv"][0])

                # OpenCV-Lab uses 0..255 with 128 as neutral for a*/b*.
                # We use these channels to prevent warm-color (red/orange/yellow) collapsing under scan/lighting shift.
                a1, b1 = float(lab[1]), float(lab[2])
                a2, b2 = float(m["lab"][1]), float(m["lab"][2])

                if sec_a == sec_b:
                    # --- RED MERGE TWEAK ---
                    # 如果兩個顏色都屬於紅色家族，允許更寬鬆的合併條件
                    if sec_a == 0:
                        dh_red = hue_diff_deg(hsv[0], m["hsv"][0])
                        de_red = de
                        if dh_red <= RED_MERGE_HUE_MAX and de_red <= RED_MERGE_DE_MAX:
                            m["cnt"] += int(cnt)
                            m["members"].append(rgb)
                            placed = True
                            break
                    # --- 原本邏輯繼續 ---
                    # Same coarse sector -> allow merge more freely (helps same-color split across the image).
                    if sec_a == 0:          # red family (wrap-around)
                        thr = 30.0
                    elif sec_a in (1, 2):   # orange / yellow
                        thr = 18.0
                    elif sec_a == 6:        # purple / magenta family (allow wider hue drift)
                        thr = 22.0
                    else:
                        thr = float(HUE_THR_DEG)

                    # Warm-color split guard (handles cases like c.jpg where orange/yellow hues are very close).
                    # Use simple R-G shape: yellow tends to have small R-G; orange tends to have larger R-G.
                    if sec_a in (1, 2):
                        rg1 = float(rgb[0]) - float(rgb[1])
                        rg2 = float(m["rgb"][0]) - float(m["rgb"][1])
                        yellowish1 = rg1 < 20.0
                        yellowish2 = rg2 < 20.0
                        # If one is yellowish and the other is orangeish, do NOT merge.
                        if yellowish1 != yellowish2:
                            continue

                    # --- NEW (v8_12 hotfix): prevent pink vs purple merging inside purple sector ---
                    # 目的：f/e 圖中，粉紅(高G、亮)與紫色(低G、較暗)不應被合併成同色。
                    if sec_a == 6:
                        g1 = int(rgb[1]); g2 = int(m["rgb"][1])
                        # both look like magenta/purple family (R & B high)
                        if (rgb[0] >= 120 and rgb[2] >= 160 and m["rgb"][0] >= 120 and m["rgb"][2] >= 160):
                            if abs(g1 - g2) >= 60:
                                continue

                    # --- NEW (v8_12 hotfix): prevent brown (dark) being merged into orange/yellow (bright) ---
                    # 目的：J 圖橘色與咖啡色差很多，別因為同一 sector 被併掉。
                    if sec_a in (1, 2):
                        v1 = float(hsv[2]); v2 = float(m["hsv"][2])
                        if abs(v1 - v2) >= 60.0 and (min(v1, v2) <= 120.0) and (max(v1, v2) >= 160.0):
                            continue

                        # 進一步的「橘 vs 咖啡」保護：兩者 Hue 很接近，
                        # 但若飽和度與亮度差距都很大，通常代表是不同顏色（例如 J 圖的橘點 vs 咖啡土）。
                        # 這個條件很保守，只在差距明顯時才阻擋合併，避免破壞一般同色陰影合併。
                        s1 = float(hsv[1]); s2 = float(m["hsv"][1])
                        if (abs(v1 - v2) >= 45.0) and (abs(s1 - s2) >= 55.0) and (min(s1, s2) <= 150.0) and (max(s1, s2) >= 185.0):
                            continue

                    if dh <= thr and de <= DE_THR_SAT:
                        m["cnt"] += int(cnt)
                        m["members"].append(rgb)
                        placed = True
                        break

                else:
                    # Cross-sector merges are risky; allow only specific adjacent cases with tight checks.

                    # Never merge red <-> yellow directly (prevents f.jpg red/yellow collapse).
                    if (sec_a == 0 and sec_b == 2) or (sec_a == 2 and sec_b == 0):
                        continue

                    pair = {sec_a, sec_b}


                    # blue <-> purple : allow merge when very close (fix purple split like (136,56,224) vs (144,48,240))
                    if pair == {5, 6}:
                        if dh <= 22.0 and de <= DE_THR_SAT:
                            m["cnt"] += int(cnt)
                            m["members"].append(rgb)
                            placed = True
                            break
                        continue
                    # red <-> orange (tight hue) : keep red shades mergeable without eating yellow
                    if pair == {0, 1}:
                        if dh <= 12.0 and de <= DE_THR_SAT:
                            m["cnt"] += int(cnt)
                            m["members"].append(rgb)
                            placed = True
                            break
                        continue

                    # orange <-> yellow : ONLY when both look "yellow-family" in Lab (high b*),
                    # so segmented yellow under lighting can merge, but orange/red won't be absorbed.
                    if pair == {1, 2}:
                        # --- NEW (v8_12 hotfix): also block dark-vs-bright merges across orange<->yellow ---
                        v1 = float(hsv[2]); v2 = float(m["hsv"][2])
                        if abs(v1 - v2) >= 60.0 and (min(v1, v2) <= 120.0) and (max(v1, v2) >= 160.0):
                            continue

                        # Only allow when BOTH look clearly yellow-ish (avoid absorbing red/orange into yellow).
                        if min(b1, b2) < 175:   # not yellowish enough
                            continue
                        if max(a1, a2) > 175:   # too red-ish in a*
                            continue
                        if min(float(hsv[0]), float(m["hsv"][0])) < 18.0:  # too close to red side
                            continue
                        if dh <= 10.0 and de <= DE_THR_SAT:
                            m["cnt"] += int(cnt)
                            m["members"].append(rgb)
                            placed = True
                            break
                        continue

                    # Other cross-sector merges are blocked
                    continue
# Case 2) both non-chromatic -> allow gray merge only
            elif (not a_ch) and (not b_ch):
                if de <= DE_THR_GRAY:
                    m["cnt"] += int(cnt)
                    m["members"].append(rgb)
                    placed = True
                    break

            # Case 3) one chromatic, one gray -> never merge
            else:
                continue

        if not placed:
            merged.append({"rgb": rgb, "cnt": int(cnt), "lab": lab, "hsv": hsv, "members": [rgb]})

    merged_counts = {m["rgb"]: m["cnt"] for m in merged}
    mapping = {}
    for m in merged:
        for o in m["members"]:
            mapping[o] = m["rgb"]
    return merged_counts, mapping


def semantic_collapse_with_mapping(color_accum, mapping):
    if not color_accum:
        return color_accum, mapping
    items = sorted(color_accum.items(), key=lambda x: x[1], reverse=True)
    total = sum(c for _, c in items) or 1
    top_rgb, top_cnt = items[0]
    if (top_cnt / total) < float(SEMANTIC_DOM):
        return color_accum, mapping

    new = {top_rgb: top_cnt}
    collapsed = set()
    for rgb, cnt in items[1:]:
        if rgb_dist(rgb, top_rgb) <= float(SEMANTIC_DIST):
            new[top_rgb] += cnt
            collapsed.add(rgb)
        else:
            new[rgb] = cnt

    new_map = {}
    for o, m in mapping.items():
        new_map[o] = top_rgb if m in collapsed else m
    return new, new_map


MIN_PROP_MERGE = 0.002
DE_THR_TINY = 60.0


def merge_tiny_into_big_with_mapping(color_accum, mapping):
    if not color_accum:
        return color_accum, mapping
    items = sorted(color_accum.items(), key=lambda x: x[1], reverse=True)
    total = sum(c for _, c in items) or 1
    big = [(rgb, cnt) for rgb, cnt in items if (cnt / total) >= MIN_PROP_MERGE]
    tiny = [(rgb, cnt) for rgb, cnt in items if (cnt / total) < MIN_PROP_MERGE]
    if not tiny or not big:
        return color_accum, mapping

    big_lab = []
    for rgb, _ in big:
        lab = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
        big_lab.append(lab)

    out = dict(big)
    merged_into = {}

    for rgb, cnt in tiny:
        lab = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
        d = [delta_e76(lab, bl) for bl in big_lab]
        j = int(np.argmin(d))
        if d[j] <= DE_THR_TINY:
            target = big[j][0]
            out[target] = out.get(target, 0) + cnt
            merged_into[rgb] = target
        else:
            out[rgb] = out.get(rgb, 0) + cnt

    new_map = {}
    for o, m in mapping.items():
        new_map[o] = merged_into.get(m, m)
    return out, new_map


def despeckle_mask(fg255: np.ndarray) -> np.ndarray:
    """Remove tiny connected components likely from scanner noise.
    Only removes components that are BOTH (area small) and (bbox small), so small digits like '2' survive.
    """
    if SPECKLE_AREA_MAX <= 0:
        return fg255
    m = (fg255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.ones(num, dtype=bool)
    keep[0] = False  # 🌟 背景不保留 (原程式這裡寫 True 但後面 out 沒畫，這裡統一改為標準寫法)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area <= int(SPECKLE_AREA_MAX) and w <= int(SPECKLE_DIM_MAX) and h <= int(SPECKLE_DIM_MAX):
            keep[i] = False
            
    # 🌟 效能優化：使用 numpy 陣列映射瞬間完成，省去 N 次迴圈全圖掃描
    return (keep[labels] * 255).astype(np.uint8)


def _cleanup_small_components(mask255: np.ndarray, area_max: int, dim_max: int) -> np.ndarray:
    """Remove tiny connected components in a binary mask.
    This is used AFTER color separation to remove scan speckles / tiny edge crumbs.
    Only removes components that are BOTH (area small) and (bbox small).
    """
    if mask255 is None:
        return mask255
    if area_max <= 0:
        return mask255
    m = (mask255 > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.ones(num, dtype=bool)
    keep[0] = False  # 🌟 背景不保留
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area <= int(area_max) and w <= int(dim_max) and h <= int(dim_max):
            keep[i] = False
            
    # 🌟 效能優化：使用 numpy 陣列映射瞬間完成
    return (keep[labels] * 255).astype(np.uint8)


def _adjacency_ratio(src255: np.ndarray, tgt255: np.ndarray, dilate_px: int = 2) -> float:
    """How much of src is adjacent to tgt (after dilation).
    Returns overlap(dilate(src), tgt) / src_pixels.
    """
    if src255 is None or tgt255 is None:
        return 0.0
    src = (src255 > 0).astype(np.uint8)
    tgt = (tgt255 > 0).astype(np.uint8)
    sp = int(src.sum())
    if sp == 0:
        return 0.0
    if dilate_px <= 0:
        dil = src
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        dil = cv2.dilate(src, k, iterations=1)
    ov = int((dil & tgt).sum())
    return float(ov) / float(sp)


def _absorb_outline_variants(arr_rgb: np.ndarray, palette: list, masks_by_color: dict) -> tuple:
    """Absorb "darker outline" small color masks into adjacent larger masks of the same hue sector.
    This reduces edge artifacts caused by strong pen pressure on borders.
    The logic is conservative to avoid breaking clean palette images (a~g).
    """
    if not palette:
        return palette, masks_by_color
    # counts per palette color
    counts = {rgb: int((masks_by_color.get(rgb, 0) > 0).sum()) for rgb, _ in palette}
    total = sum(counts.values()) or 1

    # precompute hsv/lab for palette colors
    info = {}
    for rgb in counts.keys():
        rgb_arr = np.array([[rgb]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV).reshape(3).astype(np.float32)
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
        info[rgb] = (hsv, lab, hue_sector_179(hsv[0]))

    # sort small-to-large to absorb crumbs first
    rgbs_sorted = sorted(counts.keys(), key=lambda r: counts[r])
    for src_rgb in rgbs_sorted:
        src_cnt = counts[src_rgb]
        if src_cnt == 0:
            continue
        src_prop = src_cnt / float(total)
        if src_prop > float(OUTLINE_ABSORB_MAX_PROP):
            continue  # only small-ish colors

        src_hsv, src_lab, src_sec = info[src_rgb]
        src_v = float(src_hsv[2])

        # Find best target among larger masks
        best_tgt = None
        best_score = 1e9
        for tgt_rgb, tgt_cnt in counts.items():
            if tgt_rgb == src_rgb or tgt_cnt <= src_cnt:
                continue
            tgt_hsv, tgt_lab, tgt_sec = info[tgt_rgb]
            # Prefer same hue sector (very important)
            # Allow ONLY cyan<->blue absorption with very tight bounds to handle water/sky shading.
            if tgt_sec != src_sec:
                pair = {int(src_sec), int(tgt_sec)}
                if pair != {4, 5}:  # cyan<->blue
                    continue
                dh = hue_diff_deg(src_hsv[0], tgt_hsv[0])
                if dh > 10.0:
                    continue
            # Require src is noticeably darker than tgt (outline darker)
            dv = float(tgt_hsv[2]) - src_v
            if dv < float(OUTLINE_DV_MIN):
                continue
            # Color similarity bounds
            de = delta_e76(src_lab, tgt_lab)
            if de > float(OUTLINE_DE_MAX):
                continue
            if rgb_dist(src_rgb, tgt_rgb) > float(OUTLINE_RGB_DIST_MAX):
                continue
            # Spatial adjacency
            adj = _adjacency_ratio(masks_by_color.get(src_rgb), masks_by_color.get(tgt_rgb), dilate_px=2)
            if adj < float(OUTLINE_ADJ_MIN_RATIO):
                continue

            # score: prefer tighter de, higher adjacency
            score = float(de) - 20.0 * float(adj)
            if score < best_score:
                best_score = score
                best_tgt = tgt_rgb

        if best_tgt is not None:
            # merge src into tgt
            masks_by_color[best_tgt] = cv2.bitwise_or(masks_by_color.get(best_tgt), masks_by_color.get(src_rgb))
            masks_by_color[src_rgb] = np.zeros_like(masks_by_color[best_tgt])
            # update counts/total (approx)
            counts[best_tgt] += src_cnt
            counts[src_rgb] = 0

    # Rebuild palette proportions; drop empty masks
    new_counts = {rgb: int((masks_by_color.get(rgb) > 0).sum()) for rgb in counts.keys()}
    new_total = sum(new_counts.values()) or 1
    new_palette = [(rgb, new_counts[rgb] / new_total) for rgb, _ in palette if new_counts.get(rgb, 0) > 0]
    new_palette.sort(key=lambda x: x[1], reverse=True)
    return new_palette, masks_by_color


def _black_text_candidate_mask(arr_rgb: np.ndarray, fg_bool: np.ndarray) -> np.ndarray:
    '''Build a conservative boolean mask for black text-like strokes.
    - Detect black-ish pixels in HSV within fg_bool
    - Optionally restrict to bottom-corner ROI to avoid false black on other images
    - Keep only reasonable connected components (reject tiny speckles and huge dark regions)
    Returns boolean mask.
    '''
    if fg_bool is None:
        return np.zeros(arr_rgb.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    black = (fg_bool & (hsv[..., 2] <= BLACK_V_MAX) & (hsv[..., 1] <= BLACK_S_MAX))

    if not black.any():
        return black

    # ROI restriction (bottom-left / bottom-right)
    if BLACK_TEXT_ROI_ENABLE:
        h, w = black.shape[:2]
        mx = int(max(1, round(w * float(BLACK_TEXT_ROI_X_FRAC))))
        my = int(max(1, round(h * float(BLACK_TEXT_ROI_Y_FRAC))))
        roi = np.zeros((h, w), dtype=bool)
        roi[h - my : h, :mx] = True
        roi[h - my : h, w - mx : w] = True
        black = black & roi
        if not black.any():
            return black

    # Component filtering
    m = black.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(num, dtype=bool)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox_area = bw * bh

        if area < int(BLACK_COMP_MIN_AREA):
            continue
        if bbox_area > int(BLACK_COMP_MAX_BBOX_AREA):
            continue
        if max(bw, bh) > int(BLACK_COMP_MAX_DIM):
            continue

        # reject large solid dark blobs (often shadows), allow small solid digits
        fill = area / float(bbox_area + 1e-6)
        if fill > 0.80 and bbox_area > 500:
            continue

        keep[i] = True

    if not keep.any():
        return np.zeros_like(black, dtype=bool)

    out = np.zeros_like(black, dtype=bool)
    for i in range(1, num):
        if keep[i]:
            out[labels == i] = True
    return out

def run_algorithm(arr_rgb):
    # More robust background detection for photos/scans (paper not pure white)
    bg = is_background_auto(arr_rgb)
    fg = (~bg).astype(np.uint8) * 255
    fg = despeckle_mask(fg)

    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    gid = np.full(arr_rgb.shape[:2], -1, dtype=np.int32)

    fg_bool = fg > 0
    satmask = (((hsv[..., 1] >= SAT_THR) | ((hsv[..., 1] >= SAT_THR_LIGHT) & (hsv[..., 2] >= V_LIGHT_MIN))) & fg_bool)
    bin_size = (HUE_BIN_DEG / 2.0)
    hbin = (hsv[..., 0].astype(np.float32) / bin_size).astype(np.int32)
    gid[satmask] = hbin[satmask]

    lowsat = fg_bool & (~satmask)
    if lowsat.any():
        vb = (hsv[..., 2].astype(np.int32) * GRAY_BINS // 256).astype(np.int32)
        gid[lowsat] = 1000 + vb[lowsat]

    core_only = np.full_like(arr_rgb, 255)
    color_accum = defaultdict(int)
    comps = []  # list of (dom_rgb, comp_mask_bool)

    for g in np.unique(gid[gid >= 0]):
        mask = (gid == g).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < int(MIN_STROKE_AREA):
                continue

            # 🌟 效能優化 ROI 裁切：只抓取筆畫所在的邊界框，不再掃描 300 萬像素
            x, y, w, h = stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP], stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT]
            
            roi_labels = labels[y:y+h, x:x+w]
            roi_comp = (roi_labels == lab).astype(np.uint8) * 255
            
            # 在極小的區域內算 distanceTransform，速度快上千倍
            roi_dist = cv2.distanceTransform(roi_comp, cv2.DIST_L2, 3)
            roi_core = roi_dist >= float(CORE_DIST)
            if roi_core.sum() == 0:
                roi_core = roi_comp > 0

            roi_rgb = arr_rgb[y:y+h, x:x+w]
            dom = dominant_color_mode(roi_rgb[roi_core], quant_step=max(1, QUANT_STEP))
            
            stroke_px = area
            color_accum[dom] += stroke_px
            
            # 局部更新核心畫布
            core_only[y:y+h, x:x+w][roi_core] = roi_rgb[roi_core]
            
            # 儲存區域座標與小遮罩，取代原本超級耗記憶體的大張 px_mask
            comps.append((dom, roi_comp.copy(), x, y, w, h))

    # 🌟 效能終極殺招：限制進入 O(N^2) 迴圈的顏色數量 🌟
    sorted_colors = sorted(color_accum.items(), key=lambda x: x[1], reverse=True)
    MAX_COLORS_TO_MERGE = 200 
    
    if len(sorted_colors) > MAX_COLORS_TO_MERGE:
        main_colors = dict(sorted_colors[:MAX_COLORS_TO_MERGE])
        merged_counts, mapping = semantic_merge_with_mapping(main_colors)
        
        top_1_rgb = sorted_colors[0][0]
        for rgb, cnt in sorted_colors[MAX_COLORS_TO_MERGE:]:
            mapping[rgb] = top_1_rgb
            merged_counts[top_1_rgb] = merged_counts.get(top_1_rgb, 0) + cnt
    else:
        merged_counts, mapping = semantic_merge_with_mapping(dict(color_accum))
    # ========================================================
    merged_counts, mapping = semantic_collapse_with_mapping(merged_counts, mapping)
    merged_counts, mapping = merge_tiny_into_big_with_mapping(merged_counts, mapping)
    # ---- BLACK TEXT FIX (v8_15_3): conservative black-text forcing (no extra algorithm changes) ----
    black_text_bool = None
    if FORCE_BLACK:
        black_text_bool = _black_text_candidate_mask(arr_rgb, fg_bool)
        black_cnt = int(black_text_bool.sum())
        if black_cnt >= MIN_BLACK_PIXELS:
            merged_counts[(0, 0, 0)] = max(int(merged_counts.get((0, 0, 0), 0)), black_cnt)
    # ---- NEW: 強制把「紅色」加入 palette，避免小紅色被黃/橘吸走 ----
    forced_red_rgb = None
    red_bool = None
    if FORCE_RED:
        red_bool = (fg_bool &
                    (((hsv[..., 0] <= RED_H_MAX) | (hsv[..., 0] >= RED_H_MIN))) &
                    (hsv[..., 1] >= RED_S_MIN) &
                    (hsv[..., 2] >= RED_V_MIN))
        red_cnt = int(red_bool.sum())
        if red_cnt >= MIN_RED_PIXELS:
            # 用紅色像素的平均 RGB 當作代表色（更貼近實際掃描色偏）
            rr = arr_rgb[red_bool].reshape(-1, 3).mean(axis=0)
            cand_rgb = tuple(int(round(x)) for x in rr.tolist())

            # 如果已經存在「紅色家族」的顏色，就把紅像素併到最接近的那個（避免出現兩張紅 mask）
            best_rgb = None
            best_de = 1e9
            cand_lab = cv2.cvtColor(np.array([[cand_rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
            cand_hsv = cv2.cvtColor(np.array([[cand_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV).reshape(3).astype(np.float32)

            for ex_rgb in merged_counts.keys():
                ex_hsv = cv2.cvtColor(np.array([[ex_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV).reshape(3).astype(np.float32)
                if hue_sector_179(ex_hsv[0]) != 0:
                    continue
                ex_lab = cv2.cvtColor(np.array([[ex_rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
                de_ex = float(delta_e76(cand_lab, ex_lab))
                dh_ex = float(hue_diff_deg(cand_hsv[0], ex_hsv[0]))
                if dh_ex <= 30.0 and de_ex < best_de:
                    best_de = de_ex
                    best_rgb = ex_rgb

            forced_red_rgb = best_rgb if best_rgb is not None else cand_rgb
            merged_counts[forced_red_rgb] = max(int(merged_counts.get(forced_red_rgb, 0)), 0) + red_cnt


    # ---- NEW: 濾掉「太小顏色」並把它們併回主色（不新增 mask 顏色） ----
    total_used = sum(merged_counts.values()) or 1
    items = sorted(merged_counts.items(), key=lambda kv: kv[1], reverse=True)

    # v8_12 hotfix: 如果最大宗顏色是「大面積黑背景」(例如 f 圖左右黑邊)，
    # 用它當 total 會把其他真正的筆跡顏色比例壓到太小，導致被當成 tiny 合併掉。
    # 做法：只在「判斷 tiny 顏色是否要保留」時，把黑背景從分母扣掉（但黑色仍保留輸出）。
    filter_total = total_used
    bg_black_rgb = None
    if items:
        top_rgb, top_cnt = items[0]
        top_hsv = cv2.cvtColor(np.array([[top_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV).reshape(3).astype(np.float32)
        # near-black & dominates
        if float(top_hsv[2]) <= 80.0 and (top_cnt / float(total_used)) >= 0.50:
            bg_black_rgb = top_rgb
            filter_total = max(1, int(total_used - top_cnt))

    kept = []
    for i, (rgb, cnt) in enumerate(items):
        if i == 0:
            kept.append(rgb)  # 永遠保留最大宗顏色（包含可能的黑背景）
            continue
        denom = filter_total if (bg_black_rgb is not None) else total_used
        prop = cnt / float(denom)
        # v8_12 hotfix: 保護粉紅/紫色（sector=purple），避免在 f 這種黑背景圖被比例門檻吃掉
        hsv_i = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV).reshape(3).astype(np.float32)
        if hue_sector_179(hsv_i[0]) == 6 and _is_chromatic_hsv(hsv_i) and cnt >= 120:
            kept.append(rgb)
        # v8_12 hotfix: 保護「小但很鮮豔」的顏色（例如 J 圖太陽中心的橘點）。
        # 避免被 MIN_COLOR_PIXELS 門檻直接當成 tiny 併回咖啡/黃色。
        # 這裡要求像素數 >= 120 且 S/V 都很高（一般雜訊很難滿足），因此不會爆出一堆新色。
        elif _is_chromatic_hsv(hsv_i) and (float(hsv_i[1]) >= 170.0) and (float(hsv_i[2]) >= 170.0) and (cnt >= 120):
            kept.append(rgb)
        elif (cnt >= MIN_COLOR_PIXELS) and (prop >= MIN_COLOR_PROP):
            kept.append(rgb)

    # 黑色保底：若存在黑色就強制保留
    if FORCE_BLACK and (0, 0, 0) in merged_counts and (0, 0, 0) not in kept:
        kept.append((0, 0, 0))

    # 把被濾掉的顏色重新指派到最接近的保留顏色，並且合併像素數
    if kept:
        for rgb, cnt in list(merged_counts.items()):
            if rgb in kept:
                continue
            # 重新指派到距離最近的保留色
            # 避免把真正的顏色併進「大面積黑背景」
            candidates = kept
            if bg_black_rgb is not None:
                candidates = [k for k in kept if k != bg_black_rgb] or kept
            d = [rgb_dist(rgb, k) for k in candidates]
            tgt = candidates[int(np.argmin(d))]
            mapping[rgb] = tgt
            merged_counts[tgt] = int(merged_counts.get(tgt, 0)) + int(cnt)
            merged_counts.pop(rgb, None)

    total_used = sum(merged_counts.values()) or 1
    palette = [(rgb, cnt / total_used) for rgb, cnt in merged_counts.items()]
    palette.sort(key=lambda x: x[1], reverse=True)
    palette = palette[:TOP_K]

    s = sum(p for _, p in palette) or 1.0
    palette = [(rgb, p / s) for rgb, p in palette]

    top_rgbs = [rgb for rgb, _ in palette]
    masks_by_color = {rgb: np.zeros(arr_rgb.shape[:2], dtype=np.uint8) for rgb in top_rgbs}
    # 🌟 配合 ROI，將小遮罩貼回正確的座標位置
    for dom, roi_comp, x, y, w, h in comps:
        final_rgb = mapping.get(dom, dom)
        if top_rgbs and final_rgb not in masks_by_color:
            d = [rgb_dist(final_rgb, tr) for tr in top_rgbs]
            final_rgb = top_rgbs[int(np.argmin(d))]
        if final_rgb in masks_by_color:
            masks_by_color[final_rgb][y:y+h, x:x+w][roi_comp > 0] = 255
    # ---- BLACK TEXT FIX: assign detected black-text pixels into black mask (only for those candidate pixels) ----
    if FORCE_BLACK and (black_text_bool is not None):
        black_cnt = int(black_text_bool.sum())
        if black_cnt >= MIN_BLACK_PIXELS:
            black_rgb = (0, 0, 0)
            if black_rgb not in masks_by_color:
                masks_by_color[black_rgb] = np.zeros(arr_rgb.shape[:2], dtype=np.uint8)
                palette = palette + [(black_rgb, 0.0)]
                top_rgbs = [rgb for rgb, _ in palette]
            for _rgb in list(masks_by_color.keys()):
                if _rgb != black_rgb:
                    masks_by_color[_rgb][black_text_bool] = 0
            masks_by_color[black_rgb][black_text_bool] = 255



    
    # ---- NEW: 把強制紅色像素從其他顏色移走，並塞回紅色 mask ----
    if FORCE_RED and (forced_red_rgb is not None) and (red_bool is not None):
        # 確保 palette / masks_by_color 一定包含 forced_red_rgb（就算比例很小）
        if forced_red_rgb not in masks_by_color:
            masks_by_color[forced_red_rgb] = np.zeros(arr_rgb.shape[:2], dtype=np.uint8)
            palette = palette + [(forced_red_rgb, 0.0)]
        # 從其他顏色移除紅像素
        for _rgb in list(masks_by_color.keys()):
            if _rgb != forced_red_rgb:
                masks_by_color[_rgb][red_bool] = 0
        # 加回紅色
        masks_by_color[forced_red_rgb][red_bool] = 255

    # ---- NEW: per-color tiny speckle cleanup (remove disconnected dots) ----
    # This targets small crumbs caused by scan noise / edge antialiasing.
    # Very conservative: only removes components that are BOTH small-area AND small-bbox.
    for _rgb in list(masks_by_color.keys()):
        masks_by_color[_rgb] = _cleanup_small_components(
            masks_by_color[_rgb],
            area_max=int(COLOR_SPECKLE_AREA_MAX),
            dim_max=int(COLOR_SPECKLE_DIM_MAX),
        )

    
    # ---- NEW: connect small gaps + optional opening (improves line completeness, reduces fragment masks) ----
    # 注意：這裡只對「每個顏色 mask」做形態學處理，不會影響前面的顏色取樣/分群。
    # - CLOSE：把很小的斷裂縫補起來（線條更連續）
    # - OPEN（可選）：去掉非常小的孤立噪點
    if MASK_CONNECT_ENABLE:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(MASK_CLOSE_KSIZE), int(MASK_CLOSE_KSIZE)))
        k_open = None
        if int(MASK_OPEN_KSIZE) and int(MASK_OPEN_KSIZE) > 0:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(MASK_OPEN_KSIZE), int(MASK_OPEN_KSIZE)))
        for _rgb in list(masks_by_color.keys()):
            msk = masks_by_color.get(_rgb, None)
            if msk is None:
                continue
            if (msk > 0).sum() == 0:
                continue
            # 先做 CLOSE 連接斷裂
            msk2 = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k_close, iterations=int(MASK_CLOSE_ITERS))
            # 再選擇性做 OPEN 去除孤立小點
            if k_open is not None:
                msk2 = cv2.morphologyEx(msk2, cv2.MORPH_OPEN, k_open, iterations=int(MASK_OPEN_ITERS))
            masks_by_color[_rgb] = msk2

        # 形態學後再跑一次小元件清理（避免 CLOSE 後殘留更明顯的小噪點）
        for _rgb in list(masks_by_color.keys()):
            masks_by_color[_rgb] = _cleanup_small_components(
                masks_by_color[_rgb],
                area_max=int(COLOR_SPECKLE_AREA_MAX),
                dim_max=int(COLOR_SPECKLE_DIM_MAX),
            )
# ---- NEW: absorb darker-outline variants back into main color (same sector + adjacent) ----
    # This reduces common "thick dark border" artifacts in hand-painted drawings.
    palette, masks_by_color = _absorb_outline_variants(arr_rgb, palette, masks_by_color)

    # ---- NEW: 根據最終合併完的遮罩，重新計算該遮罩在原圖上的真實代表色 ----
    new_palette = []
    new_masks_by_color = {}
    for old_rgb, prop in palette:
        mask = masks_by_color.get(old_rgb)
        if mask is None or (mask > 0).sum() == 0:
            continue
            
        # 拿著遮罩回去抓原圖 (arr_rgb) 中對應的所有像素點
        pixels = arr_rgb[mask > 0]
        if pixels.size == 0:
            continue
        
        # 重新計算這些像素點的「眾數」 (套用輕微量化 8 避免雜訊)
        new_rgb = dominant_color_mode(pixels, quant_step=8)
        
        # 防呆: 如果新計算的 RGB 已經存在，微調 B 通道避免 key 衝突
        while new_rgb in new_masks_by_color:
            new_rgb = (new_rgb[0], new_rgb[1], max(0, new_rgb[2] - 1) if new_rgb[2] > 0 else min(255, new_rgb[2] + 1))
            
        new_palette.append((new_rgb, prop))
        new_masks_by_color[new_rgb] = mask

    palette = new_palette
    masks_by_color = new_masks_by_color

    return palette, core_only, fg, masks_by_color

def save_color_masks(out_dir, palette, masks_by_color):
    for idx, (rgb, _) in enumerate(palette, 1):
        r, g, b = rgb
        m = masks_by_color.get(rgb)
        if m is None:
            continue
        cv2.imwrite(os.path.join(out_dir, MASK_BIN_FMT.format(idx=idx, r=r, g=g, b=b)), m)
        color_img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
        color_img[m > 0] = np.array([r, g, b], dtype=np.uint8)
        Image.fromarray(color_img).save(os.path.join(out_dir, MASK_COLOR_FMT.format(idx=idx, r=r, g=g, b=b)))


def list_images(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    # stable order
    files = sorted(set(files))
    return files




def morphological_skeleton(binary255: np.ndarray) -> np.ndarray:
    """Simple morphological skeletonization (no extra deps).
    Input: uint8 0/255 binary image.
    Output: uint8 0/255 skeleton.
    """
    img = (binary255 > 0).astype(np.uint8) * 255
    if img.sum() == 0:
        return img
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def write_colors_csv(out_base: str, palette: list, masks_by_color: dict):
    """Write per-image colors.csv containing RGB/HSV/Lab and final mask proportions."""
    rows = []
    # compute proportions from final masks (more faithful than early palette weights)
    pix_counts = []
    for rgb, _ in palette:
        m = masks_by_color.get(rgb)
        cnt = int((m > 0).sum()) if m is not None else 0
        pix_counts.append(cnt)
    total_pix = int(sum(pix_counts)) or 1

    for idx, ((rgb, _), cnt) in enumerate(zip(palette, pix_counts), 1):
        r, g, b = rgb
        rgb_arr = np.array([[[r, g, b]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV).reshape(3).astype(int).tolist()
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).reshape(3).astype(int).tolist()
        prop = float(cnt) / float(total_pix)
        rows.append({
            "idx": idx,
            "r": r, "g": g, "b": b,
            "h": hsv[0], "s": hsv[1], "v": hsv[2],
            "L": lab[0], "a": lab[1], "b_lab": lab[2],
            "pixels": cnt,
            "proportion": prop,
        })

    out_csv = os.path.join(out_base, "colors.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["idx","r","g","b","h","s","v","L","a","b_lab","pixels","proportion"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---- NEW: 擴充版色彩庫 (共 65 色) ----
BASIC_COLORS = {
    # 黑白灰
    (0, 0, 0): '黑色 (Black)',
    (255, 255, 255): '白色 (White)',
    (128, 128, 128): '灰色 (Gray)',
    (192, 192, 192): '銀色 (Silver)',
    (105, 105, 105): '暗灰色 (Dim Gray)',
    (211, 211, 211): '淺灰色 (Light Gray)',
    (245, 245, 245): '白煙色/極淺灰 (White Smoke)',
    
    # 紅/粉紅系
    (255, 0, 0): '紅色 (Red)',
    (139, 0, 0): '深紅色 (Dark Red)',
    (178, 34, 34): '磚紅色 (Firebrick)',
    (220, 20, 60): '猩紅色/緋紅 (Crimson)',
    (250, 128, 114): '鮭魚紅 (Salmon)',
    (255, 127, 80): '珊瑚色 (Coral)',
    (255, 192, 203): '粉紅色 (Pink)',
    (255, 105, 180): '亮粉紅色 (Hot Pink)',
    (199, 21, 133): '紫紅色 (Medium Violet Red)',
    
    # 橙/黃系
    (255, 165, 0): '橙色 (Orange)',
    (255, 140, 0): '深橙色 (Dark Orange)',
    (255, 69, 0): '橘紅色 (Orange Red)',
    (255, 215, 0): '金色 (Gold)',
    (255, 255, 0): '黃色 (Yellow)',
    (255, 250, 205): '檸檬黃 (Lemon Chiffon)',
    (240, 230, 140): '卡其色 (Khaki)',
    (238, 232, 170): '蒼白金 (Pale Goldenrod)',
    
    # 棕/土色系
    (165, 42, 42): '棕色/褐色 (Brown)',
    (139, 69, 19): '馬鞍棕色 (Saddle Brown)',
    (160, 82, 45): '赭色 (Sienna)',
    (210, 105, 30): '巧克力色 (Chocolate)',
    (205, 133, 63): '秘魯色 (Peru)',
    (222, 184, 135): '實木棕 (Burly Wood)',
    (245, 222, 179): '小麥色 (Wheat)',
    (245, 245, 220): '米色 (Beige)',
    (128, 0, 0): '栗色/酒紅 (Maroon)',
    
    # 綠色系
    (0, 255, 0): '綠色 (Green)',
    (0, 128, 0): '深綠色 (Dark Green)',
    (34, 139, 34): '森林綠 (Forest Green)',
    (128, 128, 0): '橄欖綠 (Olive)',
    (107, 142, 35): '橄欖褐 (Olive Drab)',
    (154, 205, 50): '黃綠色 (Yellow Green)',
    (144, 238, 144): '淺綠色 (Light Green)',
    (0, 255, 127): '春綠色 (Spring Green)',
    (60, 179, 113): '海藍綠 (Medium Sea Green)',
    (32, 178, 170): '淺海洋綠 (Light Sea Green)',
    
    # 青/藍色系
    (0, 255, 255): '青色/水藍 (Cyan/Aqua)',
    (0, 139, 139): '深青色 (Dark Cyan)',
    (0, 128, 128): '藍綠/水鴨色 (Teal)',
    (72, 209, 204): '綠松石色 (Turquoise)',
    (175, 238, 238): '蒼綠松石色 (Pale Turquoise)',
    (0, 0, 255): '藍色 (Blue)',
    (0, 0, 128): '深藍/海軍藍 (Navy)',
    (0, 0, 139): '暗藍色 (Dark Blue)',
    (65, 105, 225): '寶石藍 (Royal Blue)',
    (100, 149, 237): '矢車菊藍 (Cornflower Blue)',
    (30, 144, 255): '道奇藍 (Dodger Blue)',
    (135, 206, 235): '天藍色 (Sky Blue)',
    (173, 216, 230): '淺藍色 (Light Blue)',
    (70, 130, 180): '鋼青色 (Steel Blue)',
    
    # 紫/洋紅系
    (128, 0, 128): '紫色 (Purple)',
    (255, 0, 255): '洋紅 (Magenta/Fuchsia)',
    (139, 0, 139): '深洋紅色 (Dark Magenta)',
    (75, 0, 130): '靛色/深紫 (Indigo)',
    (138, 43, 226): '藍紫色 (Blue Violet)',
    (148, 0, 211): '紫羅蘭色 (Violet)',
    (153, 50, 204): '暗蘭花紫 (Dark Orchid)',
    (221, 160, 221): '梅紅色 (Plum)',
    (216, 191, 216): '薊色/淡紫 (Thistle)',
    (230, 230, 250): '薰衣草紫 (Lavender)'
}

def get_color_name(rgb):
    best_name = "未知"
    min_dist = float('inf')
    for crgb, name in BASIC_COLORS.items():
        # 計算歐式距離平方找最相近的顏色
        dist = sum((a - b)**2 for a, b in zip(rgb, crgb))
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name

# 將參數名稱改為 input_data，它可以是字串(路徑)或陣列
def process_one(input_data, out_base: str):
    os.makedirs(out_base, exist_ok=True)

    # 🌟 雙模式自動偵測
    if isinstance(input_data, str):
        # 【模式 A】單獨執行 color.py 時：傳入的是字串路徑，自己去硬碟讀圖
        img = Image.open(input_data)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
    else:
        # 【模式 B】被 features.py 呼叫時：傳入的是已經降解好的 BGR 陣列
        arr = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

    palette, core_only, fg, masks_by_color = run_algorithm(arr)
    
    # ... (下面原本的 merge_masks_by_ciede2000 等等程式碼完全不要動) ...

    # NEW: merge masks after they are created (does NOT change color detection)
    palette, masks_by_color = merge_masks_by_ciede2000(
        arr_rgb=arr,
        palette=palette,
        masks_by_color=masks_by_color,
        de_thr=19.0,
        big_prop_thr=0.30,
        small_prop_thr=0.03,
    )

    Image.fromarray(core_only).save(os.path.join(out_base, OUT_CORE))
    draw_palette_bar(palette, os.path.join(out_base, OUT_PALETTE))
    cv2.imwrite(os.path.join(out_base, OUT_BINARY), fg)
    save_color_masks(out_base, palette, masks_by_color)

    # NEW: skeleton.png (from binary strokes)
    skel = morphological_skeleton(fg)
    cv2.imwrite(os.path.join(out_base, "skeleton.png"), skel)

    # NEW: colors.csv (RGB/HSV/Lab + final mask proportion)
    write_colors_csv(out_base, palette, masks_by_color)

    print("Top colors:")
    for i, (rgb, p) in enumerate(palette, 1):
        print(f"{i:02d} RGB={rgb} proportion={p:.4f}")

    # ---- NEW: 回傳代表色給主程式寫入 CSV ----
    return palette


# ==========================================
# 供總檔 features4.py 呼叫的介面
# ==========================================
def run_color_feature(img_bgr, img_name, base_out_dir, verbose=False):    
    from pathlib import Path
    import os
    import csv

    stem = Path(img_name).stem
    color_out_dir = Path(base_out_dir) / "color" if base_out_dir else None
    
    # 建立每張圖片專屬的子資料夾
    img_out_dir = color_out_dir / stem if color_out_dir else None
    if img_out_dir:
        img_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 呼叫 color.py 裡面真正的核心函數
    # 增加一個上下文管理器來封鎖 process_one 內部的 print
    import sys
    class SuppressPrint:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    # 如果 verbose 為 False，就進入靜音模式執行演算法
    if not verbose:
        with SuppressPrint():
            pal = process_one(img_bgr, str(img_out_dir)) 
    else:
        pal = process_one(img_bgr, str(img_out_dir))

    if not pal:
        return 0, "", "", {}

    # 2. 整理總表需要的字串與準備寫入 CSV 的資料
    rgb_strs = []
    name_strs = []
    details = []
    raw_list = []
    
    for rgb, prop in pal:
        name = get_color_name(rgb) 
        hex_code = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        rgb_strs.append(hex_code)
        name_strs.append(name)
        details.append(f"{name}({rgb[0]},{rgb[1]},{rgb[2]})")
        
        raw_list.append({
            "name": name,
            "rgb": rgb,
            "prop": round(prop, 4)
        })

    color_count = len(pal)
    rgb_out = "/".join(rgb_strs)
    name_out = "/".join(name_strs)

    # 3. 獨立產出 global_colors_summary.csv
    if color_out_dir:
        csv_path = color_out_dir / "global_colors_summary.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["名稱", "顏色種類", "顏色詳情"])
            writer.writerow([stem, color_count, "、".join(details)])

    return color_count, rgb_out, name_out, {"colors": raw_list}


# ==========================================
# 單檔執行邏輯
# ==========================================
def main():
    # 🌟 1. 記錄開始時間
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Image Color Analysis")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image or directory")
    parser.add_argument("-o", "--output", required=True, help="Directory to save output files")
    args = parser.parse_args()

    input_path = args.input
    out_base = args.output
    os.makedirs(out_base, exist_ok=True)

    if os.path.isfile(input_path):
        stem = Path(input_path).stem
        img_out_dir = os.path.join(out_base, stem)
        os.makedirs(img_out_dir, exist_ok=True)
        t0 = time.time()  # 🌟 新增碼錶起點
        print(f"\n--- Processing {input_path} ---")
        pal = process_one(input_path, img_out_dir)
        print(f"  -> 此圖色彩分析耗時: {time.time() - t0:.2f} 秒")  # 🌟 新增印出耗時
        
        global_csv_path = os.path.join(out_base, "global_colors_summary.csv")
        with open(global_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["名稱", "顏色種類", "顏色詳情"])
            if pal:
                count = len(pal)
                details = []
                for rgb, prop in pal:
                    name = get_color_name(rgb)
                    details.append(f"{name}({rgb[0]},{rgb[1]},{rgb[2]})")
                writer.writerow([stem, count, "、".join(details)])
                print(f"\n已生成顏色統計表: {global_csv_path}")

    elif os.path.isdir(input_path):
        global_csv_path = os.path.join(out_base, "global_colors_summary.csv")
        with open(global_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["名稱", "顏色種類", "顏色詳情"])
            
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_exts)]
            
            for file in files:
                file_path = os.path.join(input_path, file)
                stem = Path(file).stem
                img_out_dir = os.path.join(out_base, stem)
                os.makedirs(img_out_dir, exist_ok=True)
                
                print(f"\n--- Processing {file_path} ---")
                t0 = time.time()  # 🌟 新增碼錶起點
                pal = process_one(file_path, img_out_dir)
                print(f"  -> 此圖色彩分析耗時: {time.time() - t0:.2f} 秒")  # 🌟 新增印出耗時
                
                if pal:
                    count = len(pal)
                    details = []
                    for rgb, prop in pal:
                        name = get_color_name(rgb)
                        details.append(f"{name}({rgb[0]},{rgb[1]},{rgb[2]})")
                    writer.writerow([stem, count, "、".join(details)])

        print(f"\n已生成全域顏色統計表: {global_csv_path}")

    # 🌟 2. 結算時間並印出
    end_time = time.time()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print("="*50)
    print(f"⏱️ color.py 分析結束！本次處理總共耗時: {minutes} 分 {seconds:.2f} 秒")
    print("="*50)

if __name__ == '__main__':
    main()




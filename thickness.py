# -*- coding: utf-8 -*-
"""
Step 3: 線條粗細分析 (v12 - 1752px 宇宙專用版)
加入大面積紅色塗色強制 8 分保護機制 (改採橡皮筋凸包面積法)
改名為 thickness.py 供總檔呼叫。
"""
import cv2
import numpy as np
import os
import csv
from pathlib import Path
import argparse

# ── 參數設定 ──
CONFIG = {
    # 實心區域預設粗細：設定在 L2 的 9 分區間 (9.53~11.27)
    "SOLID_DEFAULT_PX": 10.4, 
    
    # [修改] 大面積塗色保護機制
    "RED_AREA_RATIO_THRESH": 0.20,  # 設定為 20% (只要紅色圖案涵蓋紙張 20% 面積就強制給分)
    "RED_LARGE_AREA_PX": 10.4,      # 強制給予的粗細 (剛好對應 8~10分的標準區間)
}

def get_score_and_level(thk):
    """
    根據 1752px 標準計算分數與等級
    """
    # ── Level 1: 太細 (Score 1~3) ──
    if thk < 2.6:
        return 1, 1 
    elif thk < 5.2:
        return 2, 1 
    elif thk < 7.8:
        return 3, 1 
        
    # ── Level 2: 正常 (Score 8~10) ──
    elif thk < 9.53:
        return 8, 2 
    elif thk < 11.27:
        return 9, 2 
    elif thk < 13.0:
        return 10, 2 
    
    # ── Level 3: 太粗 (Score 4~7) ──
    elif thk < 14.73:
        return 4, 3 
    elif thk < 16.46:
        return 5, 3 
    elif thk < 18.19:
        return 6, 3 
    else:
        return 7, 3 

def process_image_core(struct_img, binary_img):
    """
    將核心邏輯抽離，直接吃記憶體中的圖片陣列 (BGR 的 structure 和 Gray 的 binary)
    回傳: (score, level, thickness_px, red_ratio)
    """
    # 分離顏色
    B, G, R = cv2.split(struct_img)
    green_line = (G > 50) & (R < 50)
    red_solid = (R > 150) & (G < 50)
    
    # 改用「凸包(橡皮筋)」計算紅色塗色的「實際涵蓋面積」
    total_pixels = struct_img.shape[0] * struct_img.shape[1]
    
    # 找尋畫面上所有紅色點的座標
    red_coords = cv2.findNonZero(red_solid.astype(np.uint8))
    
    # 如果有紅色點，且點的數量大於 3 (才能構成一個多邊形面積)
    if red_coords is not None and len(red_coords) > 3:
        # 使用 Convex Hull (凸包) 演算法把外圍包圍起來
        hull = cv2.convexHull(red_coords)
        red_area = cv2.contourArea(hull)
        red_ratio = red_area / total_pixels
    else:
        red_ratio = 0.0
    
    # 1. 最高優先級：如果紅色涵蓋面積大於 20% 門檻，強制給 8~10 分
    if red_ratio > CONFIG["RED_AREA_RATIO_THRESH"]:
        final_thk = CONFIG["RED_LARGE_AREA_PX"]
        
    # 2. 次優先級：計算綠色線條粗細
    elif np.any(green_line):
        dist_map = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
        thicknesses = dist_map[green_line] * 2
        final_thk = np.median(thicknesses)
        
    # 3. 最低優先級：只有極少量的紅色碎片，給預設值
    elif np.any(red_solid):
        final_thk = CONFIG["SOLID_DEFAULT_PX"]
        
    # 防呆機制 (畫面全黑沒有結構時)
    else:
        final_thk = 0.0
        
    score, level = get_score_and_level(final_thk)
    
    return score, level, round(float(final_thk), 2), red_ratio


# ==========================================
# 供單檔執行的包裝函數 (包含原版讀檔與印出)
# ==========================================
def process_single_image(struct_path):
    # 讀取 structure 圖
    struct_img = cv2.imdecode(np.fromfile(str(struct_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if struct_img is None: return None
    
    # 尋找對應的 binary 圖
    b_name = struct_path.name.replace("_3_structure", "_2_binary").replace("_structure", "_binary")
    binary_path = struct_path.parent / b_name
    
    if not binary_path.exists():
        print(f"[警告] 找不到對應的二值化圖: {binary_path.name}，跳過。")
        return None
    
    binary_img = cv2.imdecode(np.fromfile(str(binary_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if binary_img is None: return None
    
    # 呼叫核心運算
    score, level, thickness_px, red_ratio = process_image_core(struct_img, binary_img)
    
    return {
        "name": struct_path.stem.replace("_3_structure", "").replace("_structure", ""),
        "level": level,
        "score": score,
        "thickness_px": thickness_px,
        "red_ratio": red_ratio
    }

# ==========================================
# 供總檔 features4.py 呼叫的安靜介面
# ==========================================
# --- 請將以下程式碼貼在 thickness.py 最下方 ---

def run_thickness_feature(structure_mask, binary_mask, img_name, out_dir=None, verbose=False):
    from pathlib import Path
    import csv
    import os
    
    stem = Path(img_name).stem
    thick_out_dir = Path(out_dir) / "thickness" if out_dir else None
    if thick_out_dir:
        thick_out_dir.mkdir(parents=True, exist_ok=True)

    # 呼叫你 100% 原始的演算法
    score, level, thickness_px, red_ratio = process_image_core(structure_mask, binary_mask)
    
    raw_data = {
        "thickness_px": thickness_px, 
        "red_ratio": red_ratio, 
        "level": level
    }
    
    # 嚴格還原你原本單檔的 CSV 輸出格式！
    if thick_out_dir:
        csv_path = thick_out_dir / "thickness.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['name', 'level', 'score', 'thickness_px'])
            writer.writerow([stem, level, score, thickness_px])
            
    return score, raw_data
# ==========================================
# 單檔執行邏輯
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Step 3: 線條粗細分析 (支援單一檔案與資料夾)")
    parser.add_argument("-i", "--input", default="inputs", help="輸入資料夾或單一檔案路徑")    
    parser.add_argument("-o", "--output", default="output_step3", help="輸出資料夾")
    args = parser.parse_args()

    input_path = Path(args.input)
    # 輸出資料夾設定在指定目錄下的 /thickness/
    out_dir = Path(args.output) / "thickness"
    
    # 自動判斷檔案或資料夾
    files = []
    if input_path.is_file():
        if "_structure" not in input_path.name:
            print(f"[錯誤] 指定的單圖不是 _structure 圖片，無法執行。")
            return
        files = [input_path]
    elif input_path.is_dir():
        # 讀取 _structure 圖
        files = [f for f in sorted(input_path.iterdir()) if f.is_file() and "_structure" in f.name]
    else:
        print(f"[錯誤] 找不到路徑: {input_path}")
        return

    if not files:
        print("[警告] 找不到任何有效的 _structure 圖片檔案！請先執行 bina_sk。")
        return

    print(f"=== Step 3: 開始分析 {len(files)} 個項目 (1752px 標準) ===")
    
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / "thickness.csv"

    results = []
    for f in files:
        res = process_single_image(f)
        if res:
            # 存入 CSV 時把 ratio 拿掉，保持你原有的格式
            csv_data = {
                "name": res["name"],
                "level": res["level"],
                "score": res["score"],
                "thickness_px": res["thickness_px"]
            }
            results.append(csv_data)
            
            # 終端機額外顯示紅色面積比例，幫助你抓參數
            ratio_str = f"({res['red_ratio']*100:.1f}% 面積)"
            print(f"[{res['name']}] {res['thickness_px']}px {ratio_str} -> Level {res['level']} (Score {res['score']})")

    # 寫入 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['name', 'level', 'score', 'thickness_px']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in results:
            writer.writerow(data)
            
    print(f"\n[完成] 報告已生成: {csv_path}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from pathlib import Path
from skimage.morphology import skeletonize
import argparse

# ── 參數設定 ──
CONFIG = {
    # 1. 二值化 (維持 test.py 的設定，完全不動顏色)
    "SAT_TH": 40,
    "DARK_TH": 180,
    
    # 2. 結構判定參數 (維持 cuv23 / test.py 的設定)
    "min_area": 50,             
    
    # 剝洋蔥法 (判斷實心/空心)
    "erosion_iter": 8,
    "solid_ratio_th": 0.3,     
    
    "prune_length": 15,         
}

def binarize_image_fixed(img_bgr):
    """ 
    修正後的二值化：
    關鍵修改：移除 MORPH_OPEN。
    這能確保 1px 的細線（只要有顏色）不會被侵蝕掉。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    # 條件維持不變
    color_mask = S > CONFIG["SAT_TH"]
    dark_mask = V < CONFIG["DARK_TH"]
    
    fg_mask = np.logical_or(color_mask, dark_mask).astype(np.uint8) * 255
    
    # [關鍵修正] 
    # 這裡原本有 MORPH_OPEN，它是細線殺手，現在移除了。
    # 只保留 MORPH_CLOSE 用來連接斷裂的筆觸。
    kernel = np.ones((3, 3), np.uint8)
    # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel) # <--- 這一行是兇手，已註解掉
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return fg_mask

def prune_skeleton(skeleton):
    """ 骨架修剪 (維持不變) """
    skel_img = skeleton.copy()
    # 反覆修剪死路端點
    for _ in range(3):
        # 建立3x3卷積核來計算鄰居數量
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.float32)
        
        # 將骨架圖正規化到0~1並做卷積
        neighbors = cv2.filter2D(skel_img.astype(np.float32)/255.0, -1, kernel)
        
        # 找出交叉點 (中心點10 + 至少3個鄰居 = 13)
        junctions = (neighbors >= 13).astype(np.uint8) * 255
        
        # 膨脹交叉點以覆蓋周圍像素
        junctions_dilate = cv2.dilate(junctions, np.ones((3,3)))
        
        # 從骨架中移除交叉點區域，切分成獨立線段
        segments_mask = cv2.bitwise_and(skel_img, skel_img, mask=cv2.bitwise_not(junctions_dilate))
        
        # 找尋各獨立線段
        num, labels, stats, _ = cv2.connectedComponentsWithStats(segments_mask, connectivity=8)
        
        is_changed = False
        for i in range(1, num):
            # 檢查線段長度
            if stats[i, cv2.CC_STAT_AREA] < CONFIG["prune_length"]:
                # 檢查是否連接到端點 (死路)
                ys, xs = np.where(labels == i)
                coords = np.column_stack((ys, xs))
                
                is_dead_end = False
                for y, x in coords:
                    # 端點 (中心點10 + 只有1個鄰居 = 11)
                    if 10.9 < neighbors[y, x] < 11.1: 
                        is_dead_end = True
                        break
                        
                if is_dead_end:
                    skel_img[labels == i] = 0
                    is_changed = True
                    
        if not is_changed:
            break
            
    return skel_img

# --- 新增：將核心運算抽離出來，完全保留原本邏輯，方便總程式呼叫 ---
def process_single_image_core(img, stem, out_dir_path):
    # 1. 二值化 (安全版)
    binary = binarize_image_fixed(img)
    
    # 去除小雜訊
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_binary = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= CONFIG["min_area"]:
             clean_binary[labels == i] = 255

    # 2. 結構分析 (維持 cuv23 邏輯)
    num_objs, obj_labels, obj_stats, _ = cv2.connectedComponentsWithStats(clean_binary, connectivity=8)
    
    vis_structure = np.zeros_like(img)
    vis_overlay = cv2.addWeighted(img, 0.4, np.zeros_like(img), 0.6, 0)

    kernel_erode = np.ones((3,3), np.uint8)

    for i in range(1, num_objs):
        mask = (obj_labels == i).astype(np.uint8) * 255
        original_area = obj_stats[i, cv2.CC_STAT_AREA]
        
        # 判定邏輯 (剝洋蔥法)
        eroded_mask = cv2.erode(mask, kernel_erode, iterations=CONFIG["erosion_iter"])
        remaining_area = cv2.countNonZero(eroded_mask)
        ratio = remaining_area / (original_area + 1e-5)
        
        if ratio > CONFIG["solid_ratio_th"]:
            # === 模式 A: 實心區域 (紅色) ===
            # 使用 RETR_CCOMP 抓取內外輪廓
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_structure, contours, -1, (0, 0, 255), 2) 
            cv2.drawContours(vis_overlay, contours, -1, (0, 0, 255), 2)
        else:
            # === 模式 B: 線條 (綠色) ===
            skel = skeletonize(mask.astype(bool)).astype(np.uint8) * 255
            skel_pruned = prune_skeleton(skel)
            skel_vis = cv2.dilate(skel_pruned, np.ones((3,3)))
            
            vis_structure[skel_vis > 0] = [0, 255, 0]
            vis_overlay[skel_vis > 0] = [0, 255, 0]

    # 存檔 (如果有指定資料夾)
    if out_dir_path:
        out_dir = Path(out_dir_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(out_dir / f"{stem}_1_original.jpg"), img)
        cv2.imwrite(str(out_dir / f"{stem}_2_binary.png"), clean_binary)
        cv2.imwrite(str(out_dir / f"{stem}_3_structure.png"), vis_structure)
        cv2.imwrite(str(out_dir / f"{stem}_4_overlay.png"), vis_overlay)

    return clean_binary, vis_structure, vis_overlay

# --- 原本供單檔執行的函數 (保留所有原版 print) ---
def process_single_image(img_path, output_dir):
    print(f"處理: {img_path.name}")
    img = cv2.imread(str(img_path))
    if img is None: return

    # 呼叫共用核心
    process_single_image_core(img, img_path.stem, output_dir)
    
    print(f"  -> 完成")

# --- 新增：供總檔 features4.py 呼叫的安靜介面 ---
def run_binary_sk_feature(img_bgr, img_name, out_dir=None, verbose=False):
    """
    提供給總檔呼叫的介面，不亂印多餘訊息，只回傳需要的遮罩與影像。
    """
    stem = Path(img_name).stem
    if verbose:
        print(f"處理: {img_name}")
        
    clean_binary, vis_structure, vis_overlay = process_single_image_core(img_bgr, stem, out_dir)
    
    if verbose:
        print(f"  -> 完成")
        
    return clean_binary, vis_structure, vis_overlay

# ==========================================
# 單檔執行邏輯
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs", help="輸入資料夾或單一檔案")    
    parser.add_argument("--output", default="output_thin_safe", help="輸出資料夾")
    args = parser.parse_args()

    input_path = Path(args.input)
    # 依照最新要求：建立在指定輸出資料夾下，且拔掉 results 這層
    output_dir = Path(args.output) / "binary_sk"
    
    # --- 關鍵修改：自動判斷檔案或資料夾 ---
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in sorted(input_path.iterdir()) if f.is_file() and f.suffix.lower() in exts]
    else:
        print(f"[錯誤] 找不到路徑: {input_path}")
        return
    
    print(f"開始處理 {len(files)} 個項目 (移除 Open 運算以保護細線)...\n")
    for file in files:
        process_single_image(file, output_dir)
    print("\n完成！")

if __name__ == "__main__":
    main()
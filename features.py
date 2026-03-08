# features.py
# -*- coding: utf-8 -*-
"""
整合版 CV 特徵抽取：
- 僅進行排版與標註，嚴格對齊 identify.py 的序號 1~13
- 完全保留原有演算法邏輯、視覺化繪圖與除錯程式碼
- 動態性 CSV 產出完美封裝於獨立函式內，並修正計分與顯示邏輯
"""

import cv2
import numpy as np
import os
import math
from collections import defaultdict
import collections

# 匯入外部單檔模組
import bina_sk
import angle
import thickness
import fluency
import color
from ultralytics import YOLO

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

@dataclass
class FeatureResult:
    row: Dict[str, Any]
    raw: Dict[str, Any]

# =======================================================
# 基礎影像讀取與前處理工具區 (不屬於特定單一特徵)
# =======================================================
def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"無法讀取影像: {path}")
    return img

def make_drawing_mask(bgr: np.ndarray) -> np.ndarray:
    """保留給其他舊特徵使用的基礎 mask"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]
    V = hsv[..., 2]
    color_mask = S > 30
    dark_mask = V < 180
    mask = np.logical_or(color_mask, dark_mask).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def morphological_skeleton(mask_255: np.ndarray) -> np.ndarray:
    """保留給其他舊特徵使用的基礎骨架"""
    return cv2.ximgproc.thinning(mask_255)

def f_preprocess_bina_sk(bgr, img_name, base_out_dir):
    """前處理：呼叫 bina_sk 產生紅綠分析圖"""
    out_dir = os.path.join(base_out_dir, "bina_sk")
    os.makedirs(out_dir, exist_ok=True)
    binary_mask, structure_mask, overlay_mask = bina_sk.run_binary_sk_feature(
        bgr, img_name, out_dir=out_dir, verbose=False
    )
    return binary_mask, structure_mask

def _save_debug_images(image_path, bgr, mask_255, skel_255, main_cell, shadow_mask, viz_dir):
    """除錯用：儲存各種中間處理圖片"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    
    cv2.imwrite(os.path.join(viz_dir, f"{stem}_01_mask.png"), mask_255)
    cv2.imwrite(os.path.join(viz_dir, f"{stem}_02_skel.png"), skel_255)
    
    grid_img = bgr.copy()
    h, w = grid_img.shape[:2]
    for i in range(1, 3):
        cv2.line(grid_img, (0, int(i*h/3)), (w, int(i*h/3)), (255,0,0), 2)
        cv2.line(grid_img, (int(i*w/3), 0), (int(i*w/3), h), (255,0,0), 2)
    cv2.putText(grid_img, f"Main: {main_cell}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imwrite(os.path.join(viz_dir, f"{stem}_03_grid.png"), grid_img)
    
    grid_mask = cv2.cvtColor(mask_255, cv2.COLOR_GRAY2BGR)
    for i in range(1, 3):
        cv2.line(grid_mask, (0, int(i*h/3)), (w, int(i*h/3)), (255,0,0), 2)
        cv2.line(grid_mask, (int(i*w/3), 0), (int(i*w/3), h), (255,0,0), 2)
    cv2.imwrite(os.path.join(viz_dir, f"{stem}_04_grid_mask.png"), grid_mask)
    
    cv2.imwrite(os.path.join(viz_dir, f"{stem}_05_shadow.png"), shadow_mask)

# =======================================================
# 特徵函式區 (嚴格依照 identify.py 的序號 1~14 排列)
# =======================================================

# 【序號 1：放置方向 (paper_orientation)】
def f_paper_orientation(mask_255: np.ndarray) -> Tuple[int, Dict]:
    h, w = mask_255.shape
    orient = 1 if h > w else 2
    return orient, {"height": h, "width": w, "desc": "1=縱向, 2=橫向"}

# 【序號 2：弧度 (curvature)】
def f_curvature_angle(structure_mask, img_name, base_out_dir):
    return angle.run_angle_feature(structure_mask, img_name, out_dir=base_out_dir, verbose=False)

# 【序號 3：流暢度 (line_smoothness)】
def f_line_smoothness_fluency(structure_mask, img_name, base_out_dir):
    return fluency.run_fluency_feature(structure_mask, img_name, out_dir=base_out_dir, verbose=False)

# 【序號 4：粗細 (line_thickness)】
def f_line_thickness(structure_mask, binary_mask, img_name, base_out_dir):
    return thickness.run_thickness_feature(structure_mask, binary_mask, img_name, out_dir=base_out_dir, verbose=False)

# 【序號 5 & 6：顏色RGB、名稱與種類數 (colors_rgb, colors_name, color_count)】
# (註：此特徵直接於主流程呼叫 color.py 模組計算，無專屬 def 函式)

# 【序號 7 & 8：陰影面積占比與區域個數 (shadow_area_ratio, shadow_region_count)】
def f_shadow_region_count(structure_mask: np.ndarray, binary_mask: np.ndarray, img_name: str, base_out_dir: str) -> Tuple[int, int, Dict]:
    """塗色區域個數與實體面積 (紅框處理 + 二值化實體交集，完美避開甜甜圈空心並輸出實心圖)"""
    debug_dir = os.path.join(base_out_dir, "shadowarea")
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(img_name)[0]

    lower_red = np.array([0, 0, 200])
    upper_red = np.array([80, 80, 255])
    red_edges = cv2.inRange(structure_mask, lower_red, upper_red)
    
    # 填滿所有紅色邊界包圍的區域 (此時連甜甜圈洞都會先被填滿)
    h, w = red_edges.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = red_edges
    cv2.floodFill(padded, None, (0, 0), 255)
    red_holes = cv2.bitwise_not(padded[1:h+1, 1:w+1])
    filled_red_topological = cv2.bitwise_or(red_edges, red_holes)
    
    # 神奇的一步：與二值化 mask 取交集。
    final_shadow_mask = cv2.bitwise_and(filled_red_topological, binary_mask)
                
    paint_area = int(np.count_nonzero(final_shadow_mask))
    
    # 計算連通區域數量 (個數)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(final_shadow_mask, connectivity=8)
    region_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 50: # 面積大於 50 像素才算一個區域
            region_count += 1

    # 儲存成實心二值化圖
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_shadow_painted_area.png"), final_shadow_mask)
    return region_count, paint_area, {"region_count": region_count, "paint_area": paint_area}

def f_shadow_area_ratio(paint_area: int, binary_area: int) -> Tuple[float, Dict]:
    """【序號 7：陰影面積占比】塗色面積 / 純二值化實體面積"""
    ratio = float(paint_area) / float(binary_area) if binary_area > 0 else 0.0
    return round(ratio, 3), {"shadow_paint_area": paint_area, "binary_pixel_area": binary_area}

# 【序號 9：繪畫區域 (drawing_region_main, drawing_region_covered)】
def f_drawing_region_3x3(bgr: np.ndarray, pixel_mask: np.ndarray, img_name: str, base_out_dir: str) -> Tuple[Tuple[str, str], Dict]:
    h, w = pixel_mask.shape
    rh, rw = h // 3, w // 3
    
    labels = [
        ["A1", "A2", "A3"],
        ["B1", "B2", "B3"],
        ["C1", "C2", "C3"]
    ]
    
    region_counts_list = []
    region_dir = os.path.join(base_out_dir, "3x3")
    os.makedirs(region_dir, exist_ok=True)
    vis_img = bgr.copy()

    for r in range(3):
        for c in range(3):
            y1, y2 = r * rh, (r + 1) * rh
            x1, x2 = c * rw, (c + 1) * rw
            if r == 2: y2 = h
            if c == 2: x2 = w
            
            roi = pixel_mask[y1:y2, x1:x2]
            count = int(np.count_nonzero(roi))
            
            region_counts_list.append({
                "label": labels[r][c], 
                "count": count, 
                "coords": (x1, y1, x2, y2)
            })

    # 找出最大像素數量，並支援並列第一
    max_count = max(item["count"] for item in region_counts_list)
    main_items = [item for item in region_counts_list if item["count"] == max_count]
    main_region_labels = [item["label"] for item in main_items]
    main_region_str = "/".join(main_region_labels)
    
    covered_labels = [item["label"] for item in region_counts_list if item["count"] > 0]
    covered_str = "/".join(covered_labels)

    # 繪製方框
    for item in region_counts_list:
        x1, y1, x2, y2 = item["coords"]
        if item["label"] in main_region_labels:
            color = (0, 0, 255)
            thickness = 6
        else:
            color = (0, 255, 0)
            thickness = 2
            
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis_img, item["label"], (x1 + 15, y1 + 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    stem = os.path.splitext(img_name)[0]
    out_path = os.path.join(region_dir, f"{stem}_region_3x3.png")
    cv2.imencode('.png', vis_img)[1].tofile(out_path)

    raw = {
        "region_counts": {item["label"]: item["count"] for item in region_counts_list},
        "main_region": main_region_str,
        "covered_regions": covered_labels
    }
    
    return (main_region_str, covered_str), raw

# 【序號 10：繪畫面積 (drawing_area_ratio)】
def f_drawing_area_ratio(binary_mask: np.ndarray, img_name: str, base_out_dir: str) -> Tuple[float, int, int, Dict]:
    debug_dir = os.path.join(base_out_dir, "drawingarea")
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(img_name)[0]

    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    h, w = closed_mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = closed_mask
    cv2.floodFill(padded, None, (0, 0), 255)
    
    holes = cv2.bitwise_not(padded[1:h+1, 1:w+1])
    filled_drawing_mask = cv2.bitwise_or(binary_mask, holes)
        
    topo_drawing_area = int(np.count_nonzero(filled_drawing_mask))
    paper_area = binary_mask.size
    drawing_ratio = topo_drawing_area / paper_area if paper_area > 0 else 0.0

    cv2.imwrite(os.path.join(debug_dir, f"{stem}_topo_drawing_area.png"), filled_drawing_mask)
    
    return round(drawing_ratio, 3), topo_drawing_area, paper_area, {"topo_drawing_area": topo_drawing_area, "paper_area": paper_area}

# 【序號 11：筆跡強度/筆觸深淺 (stroke_depth)】
def f_stroke_depth_score(bgr: np.ndarray, mask_255: np.ndarray) -> Tuple[int, Dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask_255 > 0]
    if len(pixels) == 0:
        return 1, {"mean_gray": 255}
    
    mean_g = float(np.mean(pixels))
    
    if mean_g > 200: score = 2
    elif mean_g > 160: score = 3
    elif mean_g > 120: score = 5
    elif mean_g > 80: score = 6
    elif mean_g > 50: score = 9
    else: score = 10
        
    return score, {"mean_gray": round(mean_g, 2)}

# 【序號 12：繪畫內容 (content)】
def f_yolo_content(image_path, yolo_model, base_out_dir) -> Tuple[str, Any]:
    if yolo_model is None: 
        return "None", None
        
    content_dir = os.path.join(base_out_dir, "content")
    os.makedirs(content_dir, exist_ok=True)
    results = yolo_model.predict(source=image_path, conf=0.25, verbose=False)
    img_name = os.path.basename(image_path)
    results[0].save(filename=os.path.join(content_dir, img_name))
    
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    if len(cls_ids) == 0: 
        return "None", results
        
    counts = collections.Counter(cls_ids)
    names = yolo_model.names
    content_str = ", ".join([f"{names[i]}: {c}" for i, c in counts.items()])
    
    return content_str, results

# 【序號 13：圖案佔比 (content_size_all, content_size_paper)】
def f_content_size(results, yolo_model, paper_area) -> Tuple[str, str, Dict]:
    if results is None or len(results[0].boxes) == 0:
        return "None", "None", {}
        
    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    coords = boxes.xyxy.cpu().numpy()
    names = yolo_model.names
    
    instance_counter = collections.Counter()
    individual_areas = {} 
    total_box_area = 0.0
    
    for i, cls_id in enumerate(cls_ids):
        base_name = names[cls_id]
        instance_counter[base_name] += 1
        instance_name = f"{base_name}{instance_counter[base_name]}"
        
        x1, y1, x2, y2 = coords[i]
        area = (x2 - x1) * (y2 - y1)
        
        individual_areas[instance_name] = area
        total_box_area += area
        
    c_all_str = ", ".join([f"{name}: {a / total_box_area:.3f}" for name, a in individual_areas.items()]) if total_box_area > 0 else "None"
    c_paper_str = ", ".join([f"{name}: {a / paper_area:.3f}" for name, a in individual_areas.items()]) if paper_area > 0 else "None"
    
    raw_data = {
        "individual_areas": individual_areas,
        "total_box_area": float(total_box_area),
        "paper_area": int(paper_area)
    }
    
    return c_all_str, c_paper_str, raw_data

# 【序號 14：事物動態性 (Dynamic)】
def f_dynamic(results, yolo_model, img_name: str, base_out_dir: str) -> Tuple[Any, Dict]:
    """
    動態性判斷並將結果安靜接力寫入 CSV。
    修正：總表回傳數值(消除Excel黃色三角形)，沒辨識到內容時回傳字串 "None" 以顯示於表格。
    """
    # 1. 沒辨識到任何內容的防呆
    if results is None or len(results[0].boxes) == 0:
        dyn_score_val = "None" # 【修改】回傳字串 "None" 以顯示在表格中
        raw_data = {"dynamic_str": "None", "static_str": "None", "ratio": 0.0, "score": "None"}
    else:
        boxes = results[0].boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = yolo_model.names
        
        dynamic_classes = {"animal", "butterflies", "person"}
        static_classes = {"cloud", "flower", "grass", "house", "star", "sun", "tree"}
        
        dynamic_counter = collections.Counter()
        static_counter = collections.Counter()
        
        for cls_id in cls_ids:
            name = names[cls_id].lower()
            if name in dynamic_classes:
                dynamic_counter[name] += 1
            elif name in static_classes:
                static_counter[name] += 1
                
        dynamic_count = sum(dynamic_counter.values())
        static_count = sum(static_counter.values())
        total = dynamic_count + static_count
        
        # 用於寫入除錯 CSV 的文字明細
        dynamic_str_temp = ", ".join([f"{k}: {v}" for k, v in dynamic_counter.items()]) if dynamic_counter else "None"
        static_str_temp = ", ".join([f"{k}: {v}" for k, v in static_counter.items()]) if static_counter else "None"
        
        # 2. 有內容，但都不是動態/靜態物品，導致分母為 0 的防呆
        if total == 0:
            dyn_score_val = "None" # 【修改】回傳字串 "None" 以顯示在表格中
            raw_data = {"dynamic_str": dynamic_str_temp, "static_str": static_str_temp, "ratio": 0.0, "score": "None"}
        else:
            ratio_val = dynamic_count / total
            ratio_pct = ratio_val * 100
            
            # 【這裡只算一次分數】按照區間要求配置分數
            if ratio_pct < 10: score = 1
            elif ratio_pct < 20: score = 2
            elif ratio_pct < 30: score = 3
            elif ratio_pct < 40: score = 8
            elif ratio_pct < 50: score = 9
            elif ratio_pct < 60: score = 10
            elif ratio_pct < 70: score = 4
            elif ratio_pct < 80: score = 5
            elif ratio_pct < 90: score = 6
            else: score = 7
                
            dyn_score_val = score # 保持為整數 (int)，消滅黃色三角形
            raw_data = {
                "dynamic_str": dynamic_str_temp,
                "static_str": static_str_temp,
                "ratio": round(ratio_val, 3),
                "score": score
            }

    # 獨立寫入 CSV 邏輯
    import pandas as pd
    dyn_dir = os.path.join(base_out_dir, "dynamic")
    os.makedirs(dyn_dir, exist_ok=True)
    dyn_csv_path = os.path.join(dyn_dir, "dynamic.csv")
    
    # 建立要寫入獨立 CSV 的那一行資料
    dyn_row = [{
        "圖片名": img_name,
        "動態事物種類及數量": raw_data.get("dynamic_str", "None"),
        "靜態事物種類及數量": raw_data.get("static_str", "None"),
        "動態/靜態+動態占比": raw_data.get("ratio", 0.0),
        "事物動態性": raw_data.get("score", "None")
    }]
    
    if os.path.exists(dyn_csv_path):
        pd.DataFrame(dyn_row).to_csv(dyn_csv_path, mode='a', header=False, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(dyn_row).to_csv(dyn_csv_path, index=False, encoding="utf-8-sig")

    # 把算好的純數字或 "None" 交棒給總表
    return dyn_score_val, raw_data

# =======================================================
# 主控流程：資料字典寫入順序嚴格對齊 1~14
# =======================================================

def extract_features_for_image(image_path: str, base_out_dir: str, model=None) -> FeatureResult:
    img_name = os.path.basename(image_path)
    bgr = cv2.imread(image_path)
    if bgr is None: raise ValueError(f"無法讀取圖片: {image_path}")
        
    mask = make_drawing_mask(bgr)
    skel = morphological_skeleton(mask)

    binary_mask, structure_mask = f_preprocess_bina_sk(bgr, img_name, base_out_dir)
    success, encoded_img = cv2.imencode('.png', structure_mask)
    clean_structure_mask = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) if success else structure_mask.copy()

    # --- 預先計算 ---
    d_ratio, topo_area, paper_area, r_draw = f_drawing_area_ratio(binary_mask, img_name, base_out_dir)
    s_count, paint_area, r_shadow_cnt = f_shadow_region_count(clean_structure_mask, binary_mask, img_name, base_out_dir)
    binary_area = int(np.count_nonzero(binary_mask))
    s_ratio, r_shadow_ratio = f_shadow_area_ratio(paint_area, binary_area)
    res_tuple, r_region = f_drawing_region_3x3(bgr, binary_mask, img_name, base_out_dir)
    c_count, rgb_s, name_s, r_color = color.run_color_feature(image_path, img_name, base_out_dir, verbose=False)

    # === 呼叫 YOLO ===
    c_str, yolo_results = f_yolo_content(image_path, model, base_out_dir)
    c_all_str, c_paper_str, r_yolo_size = f_content_size(yolo_results, model, paper_area)

    # === 初始化字典 ===
    row = {}
    raw_all = {}

    # ====================================================
    # 按照表單序號 1~14 依序填入字典
    # ====================================================
    row["image"] = img_name

    v, r = f_paper_orientation(mask)
    row["paper_orientation"] = v
    raw_all["paper_orientation"] = r

    v, r = f_curvature_angle(clean_structure_mask, img_name, base_out_dir)
    row["curvature"] = v
    raw_all["curvature"] = r

    v, r = f_line_smoothness_fluency(clean_structure_mask, img_name, base_out_dir)
    row["line_smoothness"] = v
    raw_all["line_smoothness"] = r

    v, r = f_line_thickness(clean_structure_mask, binary_mask, img_name, base_out_dir)
    row["line_thickness"] = v
    raw_all["line_thickness"] = r

    row["colors_rgb"] = rgb_s
    row["colors_name"] = name_s
    row["color_count"] = c_count
    raw_all["drawing_colors"] = r_color

    row["shadow_area_ratio"] = s_ratio
    raw_all["shadow_area_ratio"] = r_shadow_ratio

    row["shadow_region_count"] = s_count
    raw_all["shadow_region_count"] = r_shadow_cnt

    row["drawing_region_main"] = res_tuple[0]
    row["drawing_region_covered"] = res_tuple[1]
    raw_all["drawing_region_3x3"] = r_region

    row["drawing_area_ratio"] = d_ratio
    raw_all["drawing_area_ratio"] = r_draw

    v, r = f_stroke_depth_score(bgr, mask)
    row["stroke_depth"] = v
    raw_all["stroke_depth"] = r

    # 【序號 12】繪畫內容 (機器學習)
    row["content"] = c_str
    raw_all["content"] = "Done" 

    # 【序號 13】圖案大小佔比 (兩種比例)
    row["content_size_all"] = c_all_str
    row["content_size_paper"] = c_paper_str
    raw_all["content_size"] = r_yolo_size

    # 【序號 14】事物動態性 (Dynamic)
    dyn_score_str, r_dyn = f_dynamic(yolo_results, model, img_name, base_out_dir)
    row["dynamic"] = dyn_score_str
    raw_all["dynamic"] = r_dyn

    # 【序號 15】情緒 (mood) - 尚未實作先佔位
    row["mood"] = None
    raw_all["mood"] = None

    # 【序號 16】文字 (word) - 尚未實作先佔位
    row["word"] = None
    raw_all["word"] = None

    # 記錄雜項資訊
    raw_all["mask"] = {"drawing_pixels": int(np.count_nonzero(mask)), "paper_pixels": int(mask.size)}
    raw_all["skeleton"] = {"skeleton_pixels": int(np.count_nonzero(skel))}

    return FeatureResult(row=row, raw=raw_all)
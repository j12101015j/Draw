# features.py
# -*- coding: utf-8 -*-
"""
整合版 CV 特徵抽取：
- 僅進行排版與標註，嚴格對齊 identify.py 的序號 1~16
- 完全保留原有演算法邏輯、視覺化繪圖與除錯程式碼
- 動態性 CSV 產出完美封裝於獨立函式內，並修正計分與顯示邏輯
- 【新增】情緒 (Mood) 與文字數字 (Word) YOLO 辨識邏輯
"""

import cv2
import numpy as np
import os
import math
import time
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

# =======================================================
# ⚙️ YOLO 信心值設定區 (可隨時調整此處以過濾雜訊)
# =======================================================
YOLO_CONF_CONTENT = 0.5   # 繪畫內容 (12, 13, 14) 的信心閾值
YOLO_CONF_MOOD    = 0.5   # 情緒 (15) 的信心閾值
YOLO_CONF_WORD    = 0.5   # 文字數字 (16) 的信心閾值
# =======================================================


@dataclass
class FeatureResult:
    row: Dict[str, Any]
    raw: Dict[str, Any]

# =======================================================
# 基礎影像讀取與前處理工具區
# =======================================================
def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"無法讀取影像: {path}")
    return img

def f_resize_image_1752(image: np.ndarray, target_size: int = 1752) -> np.ndarray:
    h, w = image.shape[:2]
    max_edge = max(h, w)
    if max_edge == target_size:
        return image
    scale = target_size / max_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized


def f_preprocess_bina_sk(bgr, img_name, base_out_dir):
    out_dir = os.path.join(base_out_dir, "bina_sk")
    os.makedirs(out_dir, exist_ok=True)
    binary_mask, structure_mask, overlay_mask = bina_sk.run_binary_sk_feature(
        bgr, img_name, out_dir=out_dir, verbose=False
    )
    return binary_mask, structure_mask

def _save_debug_images(image_path, bgr, mask_255, skel_255, main_cell, shadow_mask, viz_dir):
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
# 特徵函式區 (嚴格依照 identify.py 的序號 1~16 排列)
# =======================================================

# 【序號 1：放置方向】
def f_paper_orientation(mask_255: np.ndarray) -> Tuple[int, Dict]:
    h, w = mask_255.shape
    orient = 1 if h > w else 2
    return orient, {"height": h, "width": w, "desc": "1=縱向, 2=橫向"}

# 【序號 2：弧度】
def f_curvature_angle(structure_mask, img_name, base_out_dir):
    return angle.run_angle_feature(structure_mask, img_name, out_dir=base_out_dir, verbose=False)

# 【序號 3：流暢度】
def f_line_smoothness_fluency(structure_mask, img_name, base_out_dir):
    return fluency.run_fluency_feature(structure_mask, img_name, out_dir=base_out_dir, verbose=False)

# 【序號 4：粗細】
def f_line_thickness(structure_mask, binary_mask, img_name, base_out_dir):
    return thickness.run_thickness_feature(structure_mask, binary_mask, img_name, out_dir=base_out_dir, verbose=False)

# 【序號 7 & 8：陰影】
def f_shadow_region_count(structure_mask: np.ndarray, binary_mask: np.ndarray, img_name: str, base_out_dir: str) -> Tuple[int, int, Dict]:
    debug_dir = os.path.join(base_out_dir, "shadowarea")
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(img_name)[0]
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([80, 80, 255])
    red_edges = cv2.inRange(structure_mask, lower_red, upper_red)
    h, w = red_edges.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = red_edges
    cv2.floodFill(padded, None, (0, 0), 255)
    red_holes = cv2.bitwise_not(padded[1:h+1, 1:w+1])
    filled_red_topological = cv2.bitwise_or(red_edges, red_holes)
    final_shadow_mask = cv2.bitwise_and(filled_red_topological, binary_mask)
    paint_area = int(np.count_nonzero(final_shadow_mask))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(final_shadow_mask, connectivity=8)
    region_count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            region_count += 1
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_shadow_painted_area.png"), final_shadow_mask)
    return region_count, paint_area, {"region_count": region_count, "paint_area": paint_area}

def f_shadow_area_ratio(paint_area: int, binary_area: int) -> Tuple[float, Dict]:
    ratio = float(paint_area) / float(binary_area) if binary_area > 0 else 0.0
    return round(ratio, 3), {"shadow_paint_area": paint_area, "binary_pixel_area": binary_area}

# 【序號 9：繪畫區域】
def f_drawing_region_3x3(bgr: np.ndarray, pixel_mask: np.ndarray, img_name: str, base_out_dir: str) -> Tuple[Tuple[str, str], Dict]:
    h, w = pixel_mask.shape
    rh, rw = h // 3, w // 3
    labels = [["A1", "A2", "A3"], ["B1", "B2", "B3"], ["C1", "C2", "C3"]]
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
            region_counts_list.append({"label": labels[r][c], "count": count, "coords": (x1, y1, x2, y2)})
    max_count = max(item["count"] for item in region_counts_list)
    main_items = [item for item in region_counts_list if item["count"] == max_count]
    main_region_labels = [item["label"] for item in main_items]
    main_region_str = "/".join(main_region_labels)
    covered_labels = [item["label"] for item in region_counts_list if item["count"] > 0]
    covered_str = "/".join(covered_labels)
    for item in region_counts_list:
        x1, y1, x2, y2 = item["coords"]
        if item["label"] in main_region_labels:
            color = (0, 0, 255)
            thickness = 6
        else:
            color = (0, 255, 0)
            thickness = 2
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis_img, item["label"], (x1 + 15, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    stem = os.path.splitext(img_name)[0]
    out_path = os.path.join(region_dir, f"{stem}_region_3x3.png")
    cv2.imencode('.png', vis_img)[1].tofile(out_path)
    raw = {"region_counts": {item["label"]: item["count"] for item in region_counts_list}, "main_region": main_region_str, "covered_regions": covered_labels}
    return (main_region_str, covered_str), raw

# 【序號 10：繪畫面積】
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

# 【序號 11：筆觸深淺】
def f_stroke_depth_score(bgr: np.ndarray, mask_255: np.ndarray) -> Tuple[int, Dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask_255 > 0]
    if len(pixels) == 0: return 1, {"mean_gray": 255}
    mean_g = float(np.mean(pixels))
    if mean_g > 200: score = 2
    elif mean_g > 160: score = 3
    elif mean_g > 120: score = 5
    elif mean_g > 80: score = 6
    elif mean_g > 50: score = 9
    else: score = 10
    return score, {"mean_gray": round(mean_g, 2)}

# 【序號 12：繪畫內容 (content)】
def f_yolo_content(image_path, yolo_model, base_out_dir, bgr_resized) -> Tuple[str, Any]:
    if yolo_model is None: 
        return "None", None
    content_dir = os.path.join(base_out_dir, "content")
    os.makedirs(content_dir, exist_ok=True)
    
    # 套用上方設定的信心閾值 YOLO_CONF_CONTENT
    results = yolo_model.predict(source=bgr_resized, conf=YOLO_CONF_CONTENT, verbose=False)
    
    img_name = os.path.basename(image_path)
    results[0].save(filename=os.path.join(content_dir, img_name))
    
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    if len(cls_ids) == 0: 
        return "None", results
        
    counts = collections.Counter(cls_ids)
    names = yolo_model.names
    content_str = ", ".join([f"{names[i]}: {c}" for i, c in counts.items()])
    return content_str, results

# 【序號 13：圖案佔比】
def f_content_size(results, yolo_model, paper_area) -> Tuple[str, str, Dict]:
    if results is None or len(results[0].boxes) == 0:
        return "None", "None", {}
    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    coords = boxes.xyxy.cpu().numpy()
    names = yolo_model.names
    
    import collections
    individual_areas = collections.defaultdict(float)
    total_box_area = 0.0
    for i, cls_id in enumerate(cls_ids):
        base_name = names[cls_id]
        x1, y1, x2, y2 = coords[i]
        area = (x2 - x1) * (y2 - y1)
        individual_areas[base_name] += area
        total_box_area += area
        
    c_all_str = ", ".join([f"{name}: {a / total_box_area:.3f}" for name, a in individual_areas.items()]) if total_box_area > 0 else "None"
    c_paper_str = ", ".join([f"{name}: {a / paper_area:.3f}" for name, a in individual_areas.items()]) if paper_area > 0 else "None"
    raw_data = {"individual_areas": dict(individual_areas), "total_box_area": float(total_box_area), "paper_area": int(paper_area)}
    return c_all_str, c_paper_str, raw_data

# 【序號 14：事物動態性】
def f_dynamic(results, yolo_model, img_name: str, base_out_dir: str) -> Tuple[Any, Dict]:
    if results is None or len(results[0].boxes) == 0:
        dyn_score_val = "None"
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
            if name in dynamic_classes: dynamic_counter[name] += 1
            elif name in static_classes: static_counter[name] += 1
                
        dynamic_count = sum(dynamic_counter.values())
        static_count = sum(static_counter.values())
        total = dynamic_count + static_count
        
        dynamic_str_temp = ", ".join([f"{k}: {v}" for k, v in dynamic_counter.items()]) if dynamic_counter else "None"
        static_str_temp = ", ".join([f"{k}: {v}" for k, v in static_counter.items()]) if static_counter else "None"
        
        if total == 0:
            dyn_score_val = "None"
            raw_data = {"dynamic_str": dynamic_str_temp, "static_str": static_str_temp, "ratio": 0.0, "score": "None"}
        else:
            ratio_val = dynamic_count / total
            ratio_pct = ratio_val * 100
            
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
                
            dyn_score_val = score
            raw_data = {"dynamic_str": dynamic_str_temp, "static_str": static_str_temp, "ratio": round(ratio_val, 3), "score": score}

    import pandas as pd
    dyn_dir = os.path.join(base_out_dir, "dynamic")
    os.makedirs(dyn_dir, exist_ok=True)
    dyn_csv_path = os.path.join(dyn_dir, "dynamic.csv")
    
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

    return dyn_score_val, raw_data

# =======================================================
# 【新增】序號 15：情緒 (emotion)
# =======================================================
def f_yolo_mood(image_path, mood_model, base_out_dir, bgr_resized, img_name) -> Tuple[Any, Dict]:
    if mood_model is None: 
        return "None", {}
        
    mood_dir = os.path.join(base_out_dir, "emotion")
    os.makedirs(mood_dir, exist_ok=True)
    
    # 套用上方設定的信心閾值
    results = mood_model.predict(source=bgr_resized, conf=YOLO_CONF_MOOD, verbose=False)
    results[0].save(filename=os.path.join(mood_dir, img_name))
    
    if len(results[0].boxes) == 0:
        score = 4 # 都沒判斷到特徵
        raw_data = {"happy": 0, "unhappy": 0, "calm": 0, "score": score}
    else:
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        names = mood_model.names
        
        counts = collections.Counter([names[i].lower() for i in cls_ids])
        
        h_cnt = counts.get("happy", 0)
        u_cnt = counts.get("unhappy", 0)
        c_cnt = counts.get("calm", 0)
        
        diff = h_cnt - u_cnt
        
        # 依照你的計分邏輯
        if diff == 1: score = 8
        elif diff == 2: score = 9
        elif diff >= 3: score = 10
        elif diff == -1: score = 3
        elif diff == -2: score = 2
        elif diff <= -3: score = 1
        elif diff == 0:
            if h_cnt > 0 or u_cnt > 0:
                score = 6 # 持平
            elif c_cnt > 0:
                score = 5 # 只有 calm
            else:
                score = 4 # 防呆 (理論上不會走到這裡)
                
        raw_data = {"happy": h_cnt, "unhappy": u_cnt, "calm": c_cnt, "score": score}

    # 寫入 CSV 邏輯
    import pandas as pd
    mood_csv_path = os.path.join(mood_dir, "emotion.csv")
    mood_row = [{
        "image": img_name,
        "情緒分數": raw_data.get("score"),
        "happy": raw_data.get("happy"),
        "unhappy": raw_data.get("unhappy"),
        "calm": raw_data.get("calm")
    }]
    
    if os.path.exists(mood_csv_path):
        pd.DataFrame(mood_row).to_csv(mood_csv_path, mode='a', header=False, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(mood_row).to_csv(mood_csv_path, index=False, encoding="utf-8-sig")

    return score, raw_data

# =======================================================
# =======================================================
# 【新增】序號 16：文字 (Word / Number)
# =======================================================
def f_yolo_word(image_path, word_model, base_out_dir, bgr_resized, img_name) -> Tuple[str, Dict]:
    if word_model is None: 
        return "None", {}
        
    word_dir = os.path.join(base_out_dir, "number")
    os.makedirs(word_dir, exist_ok=True)
    
    # 套用上方設定的信心閾值
    results = word_model.predict(source=bgr_resized, conf=YOLO_CONF_WORD, verbose=False)
    results[0].save(filename=os.path.join(word_dir, img_name))
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        raw_data = {"word_str": "None"}
        _write_word_csv(img_name, "None", word_dir)
        return "None", raw_data
        
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    coords = boxes.xyxy.cpu().numpy()
    names = word_model.names
    
    # 提取所有框的資訊並由左至右排序 (x1)
    boxes_info = []
    for i, cls_id in enumerate(cls_ids):
        x1, y1, x2, y2 = coords[i]
        boxes_info.append({
            'name': names[cls_id],
            'x1': float(x1), 'y1': float(y1),
            'x2': float(x2), 'y2': float(y2),
            'cy': float((y1 + y2) / 2.0),
            'w': float(x2 - x1),
            'h': float(y2 - y1)
        })
    boxes_info.sort(key=lambda b: b['x1'])
    
    # 將靠近的數字進行群聚 (合併成二位數或三位數)
    groups = []
    for b in boxes_info:
        added = False
        for g in groups:
            last_b = g['boxes'][-1]
            
            dx = b['x1'] - last_b['x2'] # x 軸距離 (允許微重疊或些許間隔)
            dy = abs(b['cy'] - last_b['cy']) # y 軸中心點落差
            avg_w = (b['w'] + last_b['w']) / 2.0
            avg_h = (b['h'] + last_b['h']) / 2.0
            
            # 判斷邏輯：水平距離在寬度的 1.5 倍以內，且垂直沒有偏離超過高度的 0.8 倍 (兒童手寫容錯)
            if (-avg_w < dx < avg_w * 1.5) and (dy < avg_h * 0.8):
                g['boxes'].append(b)
                g['string'] += b['name']
                added = True
                break
                
        if not added:
            groups.append({'boxes': [b], 'string': b['name']})
            
    # ==========================================
    # 🌟 修改點：將組合後的文字轉為數字並由小到大排序
    # ==========================================
    words_list = [g['string'] for g in groups]
    
    # 確保字串以數字大小排列 (例如 100 會排在 69 後面，而不是 1, 100, 69)
    try:
        words_sorted = sorted(words_list, key=lambda x: int(x))
    except ValueError:
        words_sorted = sorted(words_list)
        
    # 用逗號串接 (前面一樣加個隱形 \u200B 徹底防堵 Excel 任何格式自動轉換)
    if words_sorted:
        word_str = "\u200B" + ", ".join(words_sorted)
    else:
        word_str = "None"
    
    _write_word_csv(img_name, word_str, word_dir)
    
    return word_str, {"word_str": word_str, "groups": words_sorted}

def _write_word_csv(img_name, word_str, word_dir):
    import pandas as pd
    word_csv_path = os.path.join(word_dir, "number.csv")
    word_row = [{
        "image": img_name,
        "文字": word_str
    }]
    if os.path.exists(word_csv_path):
        pd.DataFrame(word_row).to_csv(word_csv_path, mode='a', header=False, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(word_row).to_csv(word_csv_path, index=False, encoding="utf-8-sig")


# =======================================================
# 主控流程：資料字典寫入順序嚴格對齊 1~16 (無測速版)
# =======================================================
# def extract_features_for_image(image_path: str, base_out_dir: str, model=None, mood_model=None, word_model=None) -> FeatureResult:
#     img_name = os.path.basename(image_path)
#     bgr_raw = cv2.imread(image_path)
#     if bgr_raw is None: raise ValueError(f"無法讀取圖片: {image_path}")
        
#     # [步驟 1] 降解析度至1752
#     bgr = f_resize_image_1752(bgr_raw, 1752)

#     # [步驟 2] 前處理 (bina_sk.py) <-- 🌟 原本的步驟三往前推
#     binary_mask, structure_mask = f_preprocess_bina_sk(bgr, img_name, base_out_dir)
#     success, encoded_img = cv2.imencode('.png', structure_mask)
#     clean_structure_mask = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) if success else structure_mask.copy()

#     # [步驟 3] 面積、陰影與九宮格計算
#     d_ratio, topo_area, paper_area, r_draw = f_drawing_area_ratio(binary_mask, img_name, base_out_dir)
#     s_count, paint_area, r_shadow_cnt = f_shadow_region_count(clean_structure_mask, binary_mask, img_name, base_out_dir)
#     binary_area = int(np.count_nonzero(binary_mask))
#     s_ratio, r_shadow_ratio = f_shadow_area_ratio(paint_area, binary_area)
#     res_tuple, r_region = f_drawing_region_3x3(bgr, binary_mask, img_name, base_out_dir)

#     # [步驟 4] 色彩分析
#     c_count, rgb_s, name_s, r_color = color.run_color_feature(bgr, img_name, base_out_dir, verbose=False)

#     # [步驟 5] YOLO 內容與面積
#     c_str, yolo_results = f_yolo_content(image_path, model, base_out_dir, bgr)
#     c_all_str, c_paper_str, r_yolo_size = f_content_size(yolo_results, model, paper_area)

#     # =================初始化字典=================
#     row = {}; raw_all = {}
#     row["image"] = img_name
    
#     # 🌟 修改點：【序號 1】紙張方向改成吃 binary_mask
#     v, r = f_paper_orientation(binary_mask); row["paper_orientation"] = v; raw_all["paper_orientation"] = r
    
#     # [步驟 6] 弧度
#     v, r = f_curvature_angle(clean_structure_mask, img_name, base_out_dir); row["curvature"] = v; raw_all["curvature"] = r
    
#     # [步驟 7] 流暢度
#     v, r = f_line_smoothness_fluency(clean_structure_mask, img_name, base_out_dir); row["line_smoothness"] = v; raw_all["line_smoothness"] = r
    
#     # [步驟 8] 粗細
#     v, r = f_line_thickness(clean_structure_mask, binary_mask, img_name, base_out_dir); row["line_thickness"] = v; raw_all["line_thickness"] = r

#     row["colors_rgb"] = rgb_s; row["colors_name"] = name_s; row["color_count"] = c_count; raw_all["drawing_colors"] = r_color
#     row["shadow_area_ratio"] = s_ratio; raw_all["shadow_area_ratio"] = r_shadow_ratio
#     row["shadow_region_count"] = s_count; raw_all["shadow_region_count"] = r_shadow_cnt
#     row["drawing_region_main"] = res_tuple[0]; row["drawing_region_covered"] = res_tuple[1]; raw_all["drawing_region_3x3"] = r_region
#     row["drawing_area_ratio"] = d_ratio; raw_all["drawing_area_ratio"] = r_draw
    
#     # 🌟 修改點：【序號 11】繪畫力度(筆觸深淺) 改成吃 binary_mask
#     v, r = f_stroke_depth_score(bgr, binary_mask); row["stroke_depth"] = v; raw_all["stroke_depth"] = r
    
#     row["content"] = c_str; raw_all["content"] = "Done" 
#     row["content_size_all"] = c_all_str; row["content_size_paper"] = c_paper_str; raw_all["content_size"] = r_yolo_size
    
#     # [步驟 9] YOLO 動態、情緒、文字
#     dyn_score_str, r_dyn = f_dynamic(yolo_results, model, img_name, base_out_dir)
#     row["dynamic"] = dyn_score_str; raw_all["dynamic"] = r_dyn
    
#     mood_score, r_mood = f_yolo_mood(image_path, mood_model, base_out_dir, bgr, img_name)
#     row["emotion"] = mood_score; raw_all["emotion"] = r_mood
    
#     word_str, r_word = f_yolo_word(image_path, word_model, base_out_dir, bgr, img_name)
#     row["word"] = word_str; raw_all["word"] = r_word

#     # 🌟 修改點：將原始輸出的 mask 替換為 binary_mask，並徹底刪除 skeleton 的紀錄
#     raw_all["mask"] = {"drawing_pixels": int(np.count_nonzero(binary_mask)), "paper_pixels": int(binary_mask.size)}

#     return FeatureResult(row=row, raw=raw_all)


# =======================================================
# 主控流程：資料字典寫入順序嚴格對齊 1~16 (加入測速碼錶)
# =======================================================
def extract_features_for_image(image_path: str, base_out_dir: str, model=None, mood_model=None, word_model=None) -> FeatureResult:
    img_name = os.path.basename(image_path)
    
    print(f"\n--- ⏱️ 開始測量各步驟時間: {img_name} ---")
    time_log = {} # 建立碼錶紀錄本
    t_start_all = time.time()
    
    # [步驟 1] 讀圖與降解析度
    t0 = time.time()
    bgr_raw = load_image_bgr(image_path)
    if bgr_raw is None: raise ValueError(f"無法讀取圖片: {image_path}")
    bgr = f_resize_image_1752(bgr_raw, 1752)
    time_log['01_降解析度至1752'] = time.time() - t0

    # [步驟 2] 前處理 (bina_sk.py) <-- 🌟 原本的步驟三往前推
    t0 = time.time()
    binary_mask, structure_mask = f_preprocess_bina_sk(bgr, img_name, base_out_dir)
    success, encoded_img = cv2.imencode('.png', structure_mask)
    clean_structure_mask = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) if success else structure_mask.copy()
    time_log['02_前處理 (bina_sk.py)'] = time.time() - t0

    # [步驟 3] 面積、陰影與九宮格計算
    t0 = time.time()
    d_ratio, topo_area, paper_area, r_draw = f_drawing_area_ratio(binary_mask, img_name, base_out_dir)
    s_count, paint_area, r_shadow_cnt = f_shadow_region_count(clean_structure_mask, binary_mask, img_name, base_out_dir)
    binary_area = int(np.count_nonzero(binary_mask))
    s_ratio, r_shadow_ratio = f_shadow_area_ratio(paint_area, binary_area)
    res_tuple, r_region = f_drawing_region_3x3(bgr, binary_mask, img_name, base_out_dir)
    time_log['03_面積與陰影計算'] = time.time() - t0

    # [步驟 4] 色彩分析
    t0 = time.time()
    c_count, rgb_s, name_s, r_color = color.run_color_feature(bgr, img_name, base_out_dir, verbose=False)
    time_log['04_色彩分析 (color.py)'] = time.time() - t0

    # [步驟 5] YOLO 內容與面積
    t0 = time.time()
    c_str, yolo_results = f_yolo_content(image_path, model, base_out_dir, bgr)
    c_all_str, c_paper_str, r_yolo_size = f_content_size(yolo_results, model, paper_area)
    time_log['05_YOLO內容辨識'] = time.time() - t0

    # =================初始化字典=================
    row = {}; raw_all = {}
    row["image"] = img_name
    
    # 🌟 修改點：【序號 1】紙張方向改成吃 binary_mask
    v, r = f_paper_orientation(binary_mask); row["paper_orientation"] = v; raw_all["paper_orientation"] = r
    row["colors_rgb"] = rgb_s; row["colors_name"] = name_s; row["color_count"] = c_count; raw_all["drawing_colors"] = r_color
    row["shadow_area_ratio"] = s_ratio; raw_all["shadow_area_ratio"] = r_shadow_ratio
    row["shadow_region_count"] = s_count; raw_all["shadow_region_count"] = r_shadow_cnt
    row["drawing_region_main"] = res_tuple[0]; row["drawing_region_covered"] = res_tuple[1]; raw_all["drawing_region_3x3"] = r_region
    row["drawing_area_ratio"] = d_ratio; raw_all["drawing_area_ratio"] = r_draw
    
    # 🌟 修改點：【序號 11】繪畫力度(筆觸深淺) 改成吃 binary_mask
    v, r = f_stroke_depth_score(bgr, binary_mask); row["stroke_depth"] = v; raw_all["stroke_depth"] = r
    
    row["content"] = c_str; raw_all["content"] = "Done" 
    row["content_size_all"] = c_all_str; row["content_size_paper"] = c_paper_str; raw_all["content_size"] = r_yolo_size
    
    # [步驟 6] 弧度
    t0 = time.time()
    v, r = f_curvature_angle(clean_structure_mask, img_name, base_out_dir); row["curvature"] = v; raw_all["curvature"] = r
    time_log['06_弧度分析 (angle.py)'] = time.time() - t0

    # [步驟 7] 流暢度
    t0 = time.time()
    v, r = f_line_smoothness_fluency(clean_structure_mask, img_name, base_out_dir); row["line_smoothness"] = v; raw_all["line_smoothness"] = r
    time_log['07_流暢度 (fluency.py)'] = time.time() - t0

    # [步驟 8] 粗細
    t0 = time.time()
    v, r = f_line_thickness(clean_structure_mask, binary_mask, img_name, base_out_dir); row["line_thickness"] = v; raw_all["line_thickness"] = r
    time_log['08_粗細分析 (thickness.py)'] = time.time() - t0

    # [步驟 9] YOLO 動態、情緒、文字
    t0 = time.time()
    dyn_score_str, r_dyn = f_dynamic(yolo_results, model, img_name, base_out_dir)
    row["dynamic"] = dyn_score_str; raw_all["dynamic"] = r_dyn
    mood_score, r_mood = f_yolo_mood(image_path, mood_model, base_out_dir, bgr, img_name)
    row["emotion"] = mood_score; raw_all["emotion"] = r_mood
    word_str, r_word = f_yolo_word(image_path, word_model, base_out_dir, bgr, img_name)
    row["word"] = word_str; raw_all["word"] = r_word
    time_log['09_YOLO其他(動態/情緒/文字)'] = time.time() - t0

    # 🌟 修改點：將原始輸出的 mask 替換為 binary_mask，並徹底刪除 skeleton 的紀錄
    raw_all["mask"] = {"drawing_pixels": int(np.count_nonzero(binary_mask)), "paper_pixels": int(binary_mask.size)}

    # === 強制釋放巨大圖片變數 (拿掉用不到的 mask 和 skel) ===
    del bgr_raw, bgr, binary_mask, clean_structure_mask

    # 結算並印出報告
    total_t = time.time() - t_start_all
    print(f"   === 測速報告: {img_name} ===")
    for k, v in sorted(time_log.items()):
        print(f"   [{k}]: {v:.2f} 秒")
    print(f"   >>> 此圖總耗時: {total_t:.2f} 秒 <<<\n")

    return FeatureResult(row=row, raw=raw_all)
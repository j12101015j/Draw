# -*- coding: utf-8 -*-
"""
Step 6: 流暢度分析 (v17 - 斷裂像素佔比計分版)
使用「斷裂像素佔比 (break_ratio)」取代原本的斷點數量作為計分標準。
0~50% 均分給 4~10 分，50~100% 給予 1~3 分。
改名為 fluency.py 供總檔呼叫。
"""
import cv2
import numpy as np
import os
import csv
from pathlib import Path
import argparse
from skimage.morphology import skeletonize
from scipy.spatial import distance

# ── 參數設定 (針對 1752px 放大) ──
CONFIG = {
    "GAP_THRESHOLD": 45.0, 
    "MIN_LINE_LEN": 80,    
    "RED_BUFFER": 10       
}

# ── 演算法函數 ──
def get_fluency_score(break_ratio):
    if break_ratio <= 0.0714: return 10, 3
    elif break_ratio <= 0.1428: return 9, 3
    elif break_ratio <= 0.2142: return 8, 3
    elif break_ratio <= 0.2857: return 7, 2
    elif break_ratio <= 0.3571: return 6, 2
    elif break_ratio <= 0.4285: return 5, 2
    elif break_ratio <= 0.5000: return 4, 2
    elif break_ratio <= 0.6500: return 3, 1
    elif break_ratio <= 0.8000: return 2, 1
    else: return 1, 1

def find_endpoints_with_ids(skel, labels):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.float32)
    skel_norm = skel.astype(np.float32) / 255.0
    neighbors = cv2.filter2D(skel_norm, -1, kernel)
    endpoints_mask = (neighbors == 11.0)
    y_idxs, x_idxs = np.where(endpoints_mask)
    endpoints_data = []
    for y, x in zip(y_idxs, x_idxs):
        line_id = labels[y, x]
        if line_id > 0:
            endpoints_data.append(((x, y), line_id))
    return endpoints_data

def check_ink_connection(pt1, pt2, green_mask):
    mask = np.zeros_like(green_mask)
    cv2.line(mask, pt1, pt2, 255, 1)
    overlap = cv2.bitwise_and(mask, green_mask)
    line_pixels = cv2.countNonZero(mask)
    if line_pixels == 0: return False
    ink_pixels = cv2.countNonZero(overlap)
    return (ink_pixels / line_pixels) > 0.8

def get_endpoint_direction(labels, pt, line_id, trace_steps=12):
    curr = pt
    visited = {curr}
    path = [curr]
    
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    h, w = labels.shape
    
    for _ in range(trace_steps):
        next_pt = None
        for dx, dy in offsets:
            nx, ny = curr[0] + dx, curr[1] + dy
            if 0 <= nx < w and 0 <= ny < h:
                if (nx, ny) not in visited and labels[ny, nx] == line_id:
                    next_pt = (nx, ny)
                    break
        if next_pt:
            visited.add(next_pt)
            path.append(next_pt)
            curr = next_pt
        else:
            break
            
    end_pt = np.array(pt, dtype=np.float32)
    inner_pt = np.array(path[-1], dtype=np.float32)
    
    vec = end_pt - inner_pt
    norm = np.linalg.norm(vec)
    if norm > 1e-5:
        return vec / norm
    return None

def count_close_gaps(endpoints_data, green_mask, threshold, labels_final):
    if len(endpoints_data) < 2: return 0, [], set()
    coords = [data[0] for data in endpoints_data]
    ids = [data[1] for data in endpoints_data]
    points = np.array(coords)
    dist_matrix = distance.cdist(points, points, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    pairs = np.argwhere((np.triu(dist_matrix) < threshold) & (np.triu(dist_matrix) > 0))
    
    true_breaks = []
    broken_ids = set() 
    
    for i, j in pairs:
        if ids[i] == ids[j]: continue 
        pt1 = tuple(points[i])
        pt2 = tuple(points[j])
        
        vec1 = get_endpoint_direction(labels_final, pt1, ids[i], trace_steps=12)
        vec2 = get_endpoint_direction(labels_final, pt2, ids[j], trace_steps=12)
        
        if vec1 is not None and vec2 is not None:
            d12 = np.array(pt2) - np.array(pt1)
            dist12 = np.linalg.norm(d12)
            if dist12 > 1e-5:
                d12 = d12 / dist12
                dot1 = np.dot(vec1, d12)
                dot2 = np.dot(vec2, -d12)
                dot_opp = np.dot(vec1, -vec2)
                
                if dot1 < 0.5 or dot2 < 0.5 or dot_opp < 0.5:
                    continue 
        
        if not check_ink_connection(pt1, pt2, green_mask):
            true_breaks.append((pt1, pt2))
            broken_ids.add(ids[i])
            broken_ids.add(ids[j])
            
    return len(true_breaks), true_breaks, broken_ids


# ==========================================
# 核心演算法抽離 (保留 100% 原始邏輯)
# ==========================================
def process_image_core(structure, stem, out_dir_path):
    """
    接收 BGR structure 圖片陣列。
    回傳: score, level, break_count, break_ratio
    並產生跟您原本一模一樣的 fluency_debug.png
    """
    # 1. 前處理
    B, G, R = cv2.split(structure)
    red_mask = ((R > 50) & (G < 50)).astype(np.uint8) * 255
    red_dilated = cv2.dilate(red_mask, np.ones((3,3), np.uint8), iterations=CONFIG["RED_BUFFER"])
    green_mask = (G > 50) & (R < 50)
    green_ink = green_mask.astype(np.uint8) * 255
    safe_green_ink = cv2.bitwise_and(green_ink, cv2.bitwise_not(red_dilated))
    
    # 骨架化
    kernel = np.ones((3,3), np.uint8)
    connected_mask = cv2.dilate(safe_green_ink, kernel, iterations=1)
    skel = skeletonize(connected_mask.astype(bool)).astype(np.uint8) * 255
    
    # 2. 過濾
    num_labels_raw, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    clean_skel = np.zeros_like(skel)
    for i in range(1, num_labels_raw):
        if stats_raw[i, cv2.CC_STAT_AREA] >= CONFIG["MIN_LINE_LEN"]:
            clean_skel[labels_raw == i] = 255
            
    num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(clean_skel, connectivity=8)
    
    # 3. 找端點
    endpoints_data = find_endpoints_with_ids(clean_skel, labels_final)
    
    # 4. 計算斷裂與取得斷裂線段 ID
    break_count, gap_lines, broken_ids = count_close_gaps(endpoints_data, green_ink, CONFIG["GAP_THRESHOLD"], labels_final)
    
    # ==========================================
    # 5. 計算獨立分支的筆跡像素佔比
    # ==========================================
    green_total_px = cv2.countNonZero(safe_green_ink)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    red_edge_mask = np.zeros_like(red_mask)
    cv2.drawContours(red_edge_mask, red_contours, -1, 255, 1) 
    red_total_px = cv2.countNonZero(red_edge_mask)
    
    total_ink_px = green_total_px + red_total_px

    skel_norm = clean_skel.astype(np.float32) / 255.0
    kernel_neighbors = np.array([[1, 1, 1],
                                 [1, 10, 1],
                                 [1, 1, 1]], dtype=np.float32)
    neighbors_map = cv2.filter2D(skel_norm, -1, kernel_neighbors)
    junctions_mask = (neighbors_map >= 13.0) 
    
    branch_skel = clean_skel.copy()
    branch_skel[junctions_mask] = 0 
    
    _, branch_labels = cv2.connectedComponents(branch_skel, connectivity=8)
    
    broken_skel_mask = np.zeros_like(clean_skel)
    for pt1, pt2 in gap_lines:
        b_id1 = branch_labels[pt1[1], pt1[0]]
        b_id2 = branch_labels[pt2[1], pt2[0]]
        
        if b_id1 > 0: broken_skel_mask[branch_labels == b_id1] = 255
        if b_id2 > 0: broken_skel_mask[branch_labels == b_id2] = 255
            
    dilated_broken_skel = cv2.dilate(broken_skel_mask, np.ones((25, 25), np.uint8)) 
    broken_ink_mask = cv2.bitwise_and(safe_green_ink, dilated_broken_skel)
    broken_ink_px = cv2.countNonZero(broken_ink_mask)

    break_ratio = (broken_ink_px / total_ink_px) if total_ink_px > 0 else 0.0

    # ==========================================
    
    # 豁免
    if cv2.countNonZero(red_mask) > 500 and break_count == 0:
        pass 

    # 6. 評分 (改用 break_ratio 評分)
    score, level = get_fluency_score(break_ratio)
    
    # 7. 產生與原本完全一致的除錯圖
    if out_dir_path:
        h, w = clean_skel.shape
        debug_img = np.zeros((h, w, 3), dtype=np.uint8) 

        hsv_colors = np.zeros((1, num_labels_final, 3), dtype=np.uint8)
        hsv_colors[0, :, 0] = np.random.randint(0, 180, num_labels_final) 
        hsv_colors[0, :, 1] = np.random.randint(100, 180, num_labels_final) 
        hsv_colors[0, :, 2] = np.random.randint(230, 256, num_labels_final) 
        
        bgr_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)[0]
        bgr_colors[0] = [0, 0, 0] 

        debug_img = bgr_colors[labels_final]

        removed_noise = cv2.subtract(skel, clean_skel)
        debug_img[removed_noise > 0] = [120, 120, 120]

        red_border = cv2.subtract(red_dilated, red_mask)
        debug_img[red_border > 0] = [0, 0, 100]
        
        for (pt, _) in endpoints_data:
            cv2.circle(debug_img, pt, 3, (0, 255, 255), -1) 
            
        for pt1, pt2 in gap_lines:
            cv2.line(debug_img, pt1, pt2, (255, 255, 255), 2)

        out_dir = Path(out_dir_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{stem}_fluency_debug.png"
        cv2.imencode('.png', debug_img)[1].tofile(str(out_path))

    return score, level, break_count, break_ratio

# ==========================================
# 供單檔執行的包裝函數
# ==========================================
def process_single_image(struct_path, output_dir):
    structure = cv2.imdecode(np.fromfile(str(struct_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if structure is None: return None

    # 原本您的檔名處理方式
    original_name = struct_path.stem.replace("_3_structure", "").replace("_structure", "")
    
    score, level, break_count, break_ratio = process_image_core(structure, original_name, output_dir)
    
    return {
        "name": original_name,
        "fluency_level": level,
        "fluency_score": score,
        "break_count": break_count,
        "break_ratio": break_ratio  
    }

# ==========================================
# 供總檔 features4.py 呼叫的介面
# ==========================================
# --- 請將以下程式碼貼在 fluency.py 最下方 ---

def run_fluency_feature(img_structure, img_name, out_dir=None, verbose=False):
    from pathlib import Path
    import csv
    import os
    
    stem = Path(img_name).stem
    fluency_out_dir = Path(out_dir) / "fluency" if out_dir else None
    
    # 呼叫你 100% 原始的演算法
    score, level, break_count, break_ratio = process_image_core(img_structure, stem, fluency_out_dir)
    
    raw_data = {
        "break_count": break_count, 
        "break_ratio": break_ratio, 
        "fluency_level": level
    }
    
    # 嚴格還原你原本單檔的 CSV 輸出格式！
    if fluency_out_dir:
        csv_path = fluency_out_dir / "fluency.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['name', 'fluency_level', 'fluency_score', 'break_count'])
            writer.writerow([stem, level, score, break_count])
            
    return score, raw_data

# ==========================================
# 單檔主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Step 6: 流暢度分析 (斷裂佔比計分)")
    parser.add_argument("-i", "--input", default="inputs", help="輸入資料夾或單一檔案路徑")    
    parser.add_argument("-o", "--output", default="output_fluency", help="輸出資料夾")
    args = parser.parse_args()

    input_path = Path(args.input)
    # 所有輸出集中在 fluency/ 資料夾
    out_dir = Path(args.output) / "fluency"
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / "fluency.csv"

    files = []
    if input_path.is_file():
        if "_structure" not in input_path.name:
            print(f"[錯誤] 指定的單圖不是 _structure 圖片，無法執行。")
            return
        files = [input_path]
    elif input_path.is_dir():
        files = [f for f in sorted(input_path.iterdir()) if f.is_file() and "_structure" in f.name]
    else:
        print(f"[錯誤] 找不到路徑: {input_path}")
        return

    if not files:
        print("[警告] 找不到任何有效的 _structure 圖片檔案！")
        return

    print(f"=== Step 6 v17: 流暢度分析 (1752px 適配版 + 斷裂比例計分) ===\n")
    
    results = []
    for f in files:
        res = process_single_image(f, out_dir)
        if res:
            csv_data = {
                "name": res["name"],
                "fluency_level": res["fluency_level"],
                "fluency_score": res["fluency_score"],
                "break_count": res["break_count"]
            }
            results.append(csv_data)
            
            print(f"[{res['name']}] 斷點數:{res['break_count']} | 斷裂像素佔比: {res['break_ratio']:.2%} -> Sc:{res['fluency_score']}")

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['name', 'fluency_level', 'fluency_score', 'break_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in results:
            writer.writerow(data)
            
    print(f"\n[完成] 分數已依據斷裂佔比規則更新。")

if __name__ == "__main__":
    main()
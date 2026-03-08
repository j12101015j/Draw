# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import csv
from pathlib import Path
import argparse

# ── 參數設定 ──
CONFIG = {
    "corner_angle_min": 20,       
    "corner_angle_max": 150,        
    "arm_min_len": 8.0,             # 基礎最低長度 (給恐龍牙齒與階梯的特權)
    "min_dist_between_corners": 10, 
    
    "arm_dev_ratio": 0.085,    
    
    "arc_min_dev": 4.5,        
    "arc_dev_ratio": 0.09,     
}

def imread_safe(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_safe(path, img):
    try:
        is_success, im_buf_arr = cv2.imencode(Path(path).suffix, img)
        if is_success:
            im_buf_arr.tofile(path)
            return True
    except Exception:
        pass
    return False

def get_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6: return 180.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))

def get_contour_segment(pts, idx_start, idx_end):
    if idx_start <= idx_end:
        return pts[idx_start : idx_end + 1]
    else:
        return np.vstack((pts[idx_start:], pts[:idx_end + 1]))

def get_linearity(pts):
    if len(pts) < 2: return 1.0
    dist = np.linalg.norm(pts[0] - pts[-1])
    path_len = cv2.arcLength(pts, False)
    return dist / path_len if path_len > 1e-6 else 1.0

def get_max_deviation(pts):
    if len(pts) < 3: return 0.0
    p_start = pts[0].astype(np.float32)
    p_end = pts[-1].astype(np.float32)
    dist = np.linalg.norm(p_end - p_start)
    
    if dist < 1e-5:
        return np.max(np.linalg.norm(pts.astype(np.float32) - p_start, axis=1))

    line_vec = (p_end - p_start) / dist
    normal = np.array([-line_vec[1], line_vec[0]])
    
    vecs = pts.astype(np.float32) - p_start
    dists = np.abs(np.dot(vecs, normal))
    return np.max(dists)

def is_straight_arm(pts, req_len):
    dist = np.linalg.norm(pts[0] - pts[-1])
    if dist < req_len: 
        return False
    dev = get_max_deviation(pts)
    
    if dev <= 0.8:
        return True
    return dev / dist <= CONFIG["arm_dev_ratio"]

def is_arc(pts):
    if len(pts) < 5: return False
    
    dist = np.linalg.norm(pts[0] - pts[-1])
    if dist < 1e-3: return True
    
    dev = get_max_deviation(pts)
    
    if dev > CONFIG["arc_min_dev"]:
        return dev / dist > CONFIG["arc_dev_ratio"]
    return False

def calculate_score(corners, arcs):
    total = corners + arcs
    if total == 0:
        return 7
        
    ratio = corners / arcs if arcs > 0 else float('inf')
    
    if corners == 0 or ratio < (1/8):
        return 7
    elif (1/8) <= ratio < (1/6):
        return 6
    elif (1/6) <= ratio < (1/4):
        return 5
    elif (1/4) <= ratio < 0.8:
        return 4
    elif 0.8 <= ratio <= 2.0:
        if ratio <= 1.2:
            return 10
        elif ratio <= 1.6:
            return 9
        else:
            return 8
    elif 2.0 < ratio <= 3.0:
        return 3
    elif 3.0 < ratio <= 4.0:
        return 2
    else: 
        return 1

def process_single_image_core(img, stem, out_dir_path):
    """
    將原本的辨識邏輯抽離，只吃圖片與儲存路徑
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    B, G, R = cv2.split(img)
    is_red = ((R > 100) & (G < 80) & (B < 80)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    is_red_dilated = cv2.dilate(is_red, kernel, iterations=1)
    
    red_binary = cv2.bitwise_and(binary, is_red_dilated)
    red_contours, _ = cv2.findContours(red_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    analysis_binary = binary.copy()
    analysis_binary[is_red_dilated > 0] = 0
    
    contours, _ = cv2.findContours(analysis_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    all_corners = []
    all_arcs = []

    for cnt in red_contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 15: 
            continue
        
        dist = np.linalg.norm(pts[0] - pts[-1])
        if dist < 1e-3:
            all_arcs.append(pts) 
            continue
        
        dev = get_max_deviation(pts)
        if dev > 2.0 or (dev / dist) > 0.05: 
            all_arcs.append(pts)

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 15:
            continue

        epsilon = cv2.arcLength(pts, True) * 0.012
        epsilon = max(min(epsilon, 5.0), 1.5)
        approx = cv2.approxPolyDP(pts, epsilon, True)

        num_ap = len(approx)
        if num_ap < 3:
            if is_arc(pts):
                all_arcs.append(pts)
            continue

        approx_pts = approx.reshape(-1, 2)
        
        approx_idx = []
        for ap in approx_pts:
            dists = np.sum((pts - ap)**2, axis=1)
            approx_idx.append(np.argmin(dists))

        corner_indices = []
        for i in range(num_ap):
            idx_prev = approx_idx[i - 1]
            idx_curr = approx_idx[i]
            idx_next = approx_idx[(i + 1) % num_ap]

            p_prev = pts[idx_prev]
            p_curr = pts[idx_curr]
            p_next = pts[idx_next]

            angle = get_angle(p_prev, p_curr, p_next)

            if angle < CONFIG["corner_angle_min"] or angle > CONFIG["corner_angle_max"]:
                continue

            req_len = CONFIG["arm_min_len"] 
            
            if angle > 130:
                req_len = 18.0 
            elif angle > 110:
                req_len = 12.0

            arm1_pts = get_contour_segment(pts, idx_prev, idx_curr)
            arm2_pts = get_contour_segment(pts, idx_curr, idx_next)
            
            if is_straight_arm(arm1_pts, req_len) and is_straight_arm(arm2_pts, req_len):
                all_corners.append(tuple(p_curr))
                corner_indices.append(idx_curr)

        if not corner_indices:
            if is_arc(pts):
                all_arcs.append(pts)
        else:
            corner_indices = sorted(list(set(corner_indices)))
            for i in range(len(corner_indices)):
                idx_start = corner_indices[i]
                idx_end = corner_indices[(i+1) % len(corner_indices)]
                seg = get_contour_segment(pts, idx_start, idx_end)
                if is_arc(seg):
                    all_arcs.append(seg)

    final_corners = []
    if all_corners:
        pts_arr = np.array(all_corners)
        keep = np.ones(len(pts_arr), dtype=bool)
        for i in range(len(pts_arr)):
            if not keep[i]: continue
            for j in range(i+1, len(pts_arr)):
                if np.linalg.norm(pts_arr[i] - pts_arr[j]) < CONFIG["min_dist_between_corners"]:
                    keep[j] = False
        final_corners = pts_arr[keep]

    score = calculate_score(len(final_corners), len(all_arcs))

    vis = np.zeros_like(img)
    vis[binary > 0] = [255, 144, 30]

    for seg in all_arcs:
        pts_reshape = seg.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts_reshape], False, (255, 255, 0), 2)

    for pt in final_corners:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

    # 如果有給路徑就存檔
    if out_dir_path:
        out_dir = Path(out_dir_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{stem}_analysis.png"
        imwrite_safe(str(out_path), vis)

    return score, len(final_corners), len(all_arcs), vis

# ==========================================
# 供單檔執行的包裝函數 (包含原版 print)
# ==========================================
def process_match_image(img_path, output_dir):
    img = imread_safe(str(img_path))
    if img is None: return None
    
    stem = img_path.stem
    # 單檔邏輯：輸出到 指定輸出路徑/angle/ (不再建立各圖片資料夾)
    img_out_dir = Path(output_dir)
    
    score, corner_cnt, arc_cnt, vis = process_single_image_core(img, stem, img_out_dir)
    
    print(f"處理: {img_path.name}")
    print(f"  -> 偵測到: {corner_cnt} 個角點, {arc_cnt} 段弧線, 分數: {score}")
    
    return (img_path.name, score)

# ==========================================
# 供總檔 features4.py 呼叫的介面
# ==========================================
# --- 請將以下程式碼貼在 angle.py 最下方 ---

def run_angle_feature(img_structure, img_name, out_dir=None, verbose=False):
    from pathlib import Path
    import csv
    import os
    
    stem = Path(img_name).stem
    angle_out_dir = Path(out_dir) / "angle" if out_dir else None
    if angle_out_dir:
        angle_out_dir.mkdir(parents=True, exist_ok=True)
        
    score, corner_cnt, arc_cnt, vis = process_single_image_core(img_structure, stem, angle_out_dir)
    
    raw_data = {
        "corners_count": corner_cnt, 
        "arcs_count": arc_cnt
    }
    
    # 嚴格還原你原本單檔的 CSV 輸出格式！
    if angle_out_dir:
        csv_path = angle_out_dir / "angle.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["圖片名稱", "分數"])
            writer.writerow([f"{stem}_3_structure.png", score]) 
            
    return score, raw_data

# ==========================================
# 單檔執行主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="inputs", help="輸入資料夾或單一檔案")
    parser.add_argument("-o", "--output", default="output_match", help="輸出資料夾")
    args = parser.parse_args()

    input_path = Path(args.input)
    # 單檔執行時的根目錄： output/angle/
    output_root = Path(args.output) / "angle"
    
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
        print("[錯誤] 資料夾內找不到 _structure 結尾的圖片！請先執行 bina_sk 產生前處理圖片。")
        return
        
    print(f"開始分析 {len(files)} 個結構項目...\n")
    
    results = []
    for file in files:
        res = process_match_image(file, output_root)
        if res is not None:
            results.append(res)
            
    # 將總分數 CSV 輸出到 angle 根目錄，並改名為 angle.csv
    output_root.mkdir(exist_ok=True, parents=True)
    csv_path = output_root / "angle.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["圖片名稱", "分數"])
        writer.writerows(results)
        
    print(f"\n分析完成！分數已儲存至: {csv_path}")

if __name__ == "__main__":
    main()
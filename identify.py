# identify.py
# -*- coding: utf-8 -*-
"""
主程式：整合式影像特徵提取工具
功能：支援單張圖片或資料夾批量處理，將 CV 特徵與 YOLO 內容辨識整合輸出至總表。
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from features import extract_features_for_image

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description="Extract CV features (Integrated)")
    parser.add_argument("--input", required=True, help="輸入圖片資料夾或單一圖片路徑")
    parser.add_argument("--output", required=True, help="輸出結果資料夾名稱")
    args = parser.parse_args()

    input_dir = args.input
    base_out_dir = args.output  

    os.makedirs(base_out_dir, exist_ok=True)
    
    # =======================================================
    # 新增：大掃除，開始前自動清除舊的動態性 CSV，避免重複疊加
    # =======================================================
    dyn_csv_path = os.path.join(base_out_dir, "dynamic", "dynamic.csv")
    if os.path.exists(dyn_csv_path):
        try: os.remove(dyn_csv_path)
        except: pass

    # YOLO 模型路徑
    #model_path = r"E:\XTX2\kidsdrawing.v3i.yolov8\內容辨識\model\merged_10_classes_v8s_oversample_OK\weights\best.pt"
    model_path="models/content.pt"
    print(f"[INFO] 正在載入 YOLO 模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[警告] 找不到 YOLO 模型: {model_path}，將無法執行內容辨識。")
        yolo_model = None
    else:
        yolo_model = YOLO(model_path)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    
    if os.path.isfile(input_dir):
        images = [os.path.basename(input_dir)]
        input_parent = os.path.dirname(input_dir)
    else:
        images = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        images.sort()
        input_parent = input_dir

    if not images:
        print(f"[警告] 找不到有效的圖片檔。路徑: {input_dir}")
        return

    print(f"========================================")
    print(f"開始處理，共找到 {len(images)} 張圖片")
    print(f"輸出主資料夾設定為: {base_out_dir}")
    print(f"狀態: 正在執行特徵分析，請稍候...")
    print(f"========================================")

    rows = []
    rows_cn = [] # 中文版專屬列表
    raw_json = {}

    for img_name in images:
        img_path = os.path.join(input_parent, img_name)
        try:
            print(f"-> 正在分析: {img_name}")
            # 注意：請確認你的 features.py 接收參數的方式是 yolo_model 還是 model
            # 若為 model，請把 yolo_model=yolo_model 改為 model=yolo_model
            res = extract_features_for_image(img_path, base_out_dir, model=yolo_model)
            
            if res is None:
                continue

            # 1. 原版英文存入
            rows.append(res.row)
            
            # 2. 新增：中文版欄位直接對應存入
            row_cn = {
                "圖檔名稱": res.row.get("image"),
                "紙張方向": res.row.get("paper_orientation"),
                "畫面弧度(分)": res.row.get("curvature"),
                "線條流暢度(分)": res.row.get("line_smoothness"),
                "線條粗細(分)": res.row.get("line_thickness"),
                "使用顏色(RGB)": res.row.get("colors_rgb"),
                "使用顏色(名稱)": res.row.get("colors_name"),
                "顏色數量(種)": res.row.get("color_count"),
                "陰影區域占繪畫比": res.row.get("shadow_area_ratio"),
                "陰影區域(個)": res.row.get("shadow_region_count"),
                "主要繪畫區域": res.row.get("drawing_region_main"),
                "繪畫涵蓋區域": res.row.get("drawing_region_covered"),
                "繪畫占紙張比": res.row.get("drawing_area_ratio"),
                "繪畫力度": res.row.get("stroke_depth"),
                "繪畫內容及數量": res.row.get("content"),
                "繪畫物品占繪畫內容比": res.row.get("content_size_all"),
                "繪畫物品占紙張比": res.row.get("content_size_paper"),
                "事物動態性": res.row.get("dynamic"),
                "情緒": res.row.get("mood"),
                "文字": res.row.get("word")
            }
            rows_cn.append(row_cn)
            
            raw_json[img_name] = res.raw
            print(f"[OK] 分析完成")
        except Exception as e:
            print(f"[ERROR] 處理失敗 {img_name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df_cn = pd.DataFrame(rows_cn)

        out_csv = os.path.join(base_out_dir, "features.csv")
        out_csv_cn = os.path.join(base_out_dir, "features_CN.csv") # 中文 CSV
        out_xlsx = os.path.join(base_out_dir, "features.xlsx")
        out_xlsx_cn = os.path.join(base_out_dir, "features_CN.xlsx") # 中文 XLSX
        out_json = os.path.join(base_out_dir, "features.json")

        # 寫入 CSV
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        df_cn.to_csv(out_csv_cn, index=False, encoding="utf-8-sig")
        print(f"[INFO] 總表 CSV (含中文版) 已儲存至 {out_csv} 與 {out_csv_cn}")
        
        # 寫入 Excel
        try:
            with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Features')
            with pd.ExcelWriter(out_xlsx_cn, engine='openpyxl') as writer:
                df_cn.to_excel(writer, index=False, sheet_name='Features')
            print(f"[INFO] 總表 XLSX (含中文版) 已儲存至 {out_xlsx} 與 {out_xlsx_cn}")
        except ModuleNotFoundError:
            df.to_excel(out_xlsx, index=False)
            df_cn.to_excel(out_xlsx_cn, index=False)
            print(f"[INFO] 總表 XLSX (含中文版) 已儲存 (提示：安裝 openpyxl 可啟用自動欄寬)")

        # 寫入 JSON
        if raw_json:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(raw_json, f, indent=4, ensure_ascii=False, cls=NpEncoder)
            print(f"[INFO] 總表 JSON 已儲存至 {out_json}")
        
        print(f"========================================")
    else:
        print("[警告] 無法生成總表。")

if __name__ == "__main__":
    main()
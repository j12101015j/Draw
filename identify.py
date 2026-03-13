# identify.py
# -*- coding: utf-8 -*-
"""
主程式：整合式影像特徵提取工具
功能：支援單張圖片或資料夾批量處理，將 CV 特徵與 YOLO 內容辨識整合輸出至總表。
"""

import argparse
import time
import torch
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from features import extract_features_for_image

# =======================================================
# ⚙️ 系統控制區 (在這裡手動更改設定)
# =======================================================
FORCE_DEVICE = "cpu"  # 填入 "auto" (自動偵測), "cpu" (強制使用 CPU), 或 "gpu" (強制使用 GPU)
# =======================================================

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
    
    # 直接讀取控制區的設定，並轉成小寫防呆
    device_choice = FORCE_DEVICE.lower()

    # 開始計時
    start_time = time.time()

    # =======================================================
    # 決定 YOLO 運算設備 (CPU 或 GPU)
    # =======================================================
    has_gpu = torch.cuda.is_available()
    if device_choice == "gpu" and not has_gpu:
        print("[系統] ⚠️ 警告：您選擇了 GPU，但系統未偵測到可用的 CUDA 環境，將強制降級使用 CPU。")
        actual_device = "cpu"
    elif device_choice == "gpu" and has_gpu:
        actual_device = "cuda:0"
    elif device_choice == "cpu":
        actual_device = "cpu"
    else: # auto
        actual_device = "cuda:0" if has_gpu else "cpu"

    print("="*50)
    if actual_device == "cpu":
        print("[系統] 🐢 本次 YOLO 辨識將使用「純 CPU」運算。")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[系統] 🚀 本次 YOLO 辨識已啟用「GPU 加速」: {gpu_name} ({actual_device})")
    print("="*50)

    os.makedirs(base_out_dir, exist_ok=True)
    
    # =======================================================
    # 大掃除：開始前自動清除舊的各項 CSV，避免接力寫入時重複疊加
    # =======================================================
    csv_to_clear = [
        os.path.join(base_out_dir, "dynamic", "dynamic.csv"),
        os.path.join(base_out_dir, "emotion", "emotion.csv"),
        os.path.join(base_out_dir, "number", "number.csv")
    ]
    for csv_path in csv_to_clear:
        if os.path.exists(csv_path):
            try: os.remove(csv_path)
            except: pass

    # =======================================================
    # 載入 YOLO 模型 (共 3 個)
    # =======================================================
    # 1. 內容模型 (Content)
    model_path_content = "models/content.pt"
    print(f"[INFO] 正在載入 YOLO 內容模型: {model_path_content}")
    yolo_model_content = YOLO(model_path_content) if os.path.exists(model_path_content) else None
    if yolo_model_content: 
        yolo_model_content.to(actual_device)
    else: 
        print("[警告] 找不到內容模型。")

    # 2. 情緒模型 (Mood / Emotion)
    model_path_mood = "models/emotion.pt"
    print(f"[INFO] 正在載入 YOLO 情緒模型: {model_path_mood}")
    yolo_model_mood = YOLO(model_path_mood) if os.path.exists(model_path_mood) else None
    if yolo_model_mood: 
        yolo_model_mood.to(actual_device)
    else: 
        print("[警告] 找不到情緒模型。")

    # 3. 文字模型 (Word / Number)
    model_path_word = "models/pure_draw_6_1_best_v8s.pt"
    print(f"[INFO] 正在載入 YOLO 文字模型: {model_path_word}")
    yolo_model_word = YOLO(model_path_word) if os.path.exists(model_path_word) else None
    if yolo_model_word: 
        yolo_model_word.to(actual_device)
    else: 
        print("[警告] 找不到文字模型。")

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
            
            # 傳入 3 個不同的模型給 features
            res = extract_features_for_image(
                img_path, 
                base_out_dir, 
                model=yolo_model_content,
                mood_model=yolo_model_mood,
                word_model=yolo_model_word
            )
            
            if res is None:
                continue

            # 1. 原版英文存入
            rows.append(res.row)
            
            # 2. 中文版欄位直接對應存入
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
                "情緒": res.row.get("emotion"),
                "文字": res.row.get("word")
            }
            rows_cn.append(row_cn)
            
            #raw_json[img_name] = res.raw
            #raw_json[img_name] = res.row  # 改成 res.row，就會跟 features.csv 裡面的英文欄位一模一樣！
            raw_json[img_name] = row_cn   # 改成 row_cn，就會吐出帶有中文 Key 的 JSON！
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
            from openpyxl.utils import get_column_letter
            
            # 定義一個自動調整欄寬的小工具
            def autofit_columns(writer, dataframe, sheet_name):
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(dataframe.columns, 1):
                    col_letter = get_column_letter(idx)
                    # 找出該欄位內容或標題的最長字元數
                    max_len = max(dataframe.iloc[:, idx-1].astype(str).map(len).max(), len(str(col)))
                    # 因為中文字比較寬，所以乘上一個係數加上緩衝空間
                    worksheet.column_dimensions[col_letter].width = max_len * 1.8 + 2

            with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Features')
                autofit_columns(writer, df, 'Features') # 🌟 呼叫自動欄寬
                
            with pd.ExcelWriter(out_xlsx_cn, engine='openpyxl') as writer:
                df_cn.to_excel(writer, index=False, sheet_name='Features')
                autofit_columns(writer, df_cn, 'Features') # 🌟 呼叫自動欄寬
                
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
        
    else:
        print("[警告] 無法生成總表。")

    # 結算總時間
    end_time = time.time()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print("="*50)
    print(f"⏱️ 任務結束！本次處理總共耗時: {minutes} 分 {seconds:.2f} 秒")
    print("="*50)

if __name__ == "__main__":
    main()
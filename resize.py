# -*- coding: utf-8 -*-
"""
批次圖片縮放工具 (等比例縮放，保護特徵不變形)
"""
import cv2
import numpy as np
import argparse
from pathlib import Path

# ==========================================
# ⚙️ 參數設定區 (您可以直接在這裡改預設數字)
# ==========================================
DEFAULT_SIZE = 1752  # 預設將最長邊縮放至 1500 像素  約a4一半
# ==========================================

def resize_image_keep_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    max_edge = max(h, w)
    
    # 如果本來就一樣大，就不處理
    if max_edge == target_size:
        return image
        
    # 計算縮放比例
    scale = target_size / max_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 電腦視覺小訣竅：縮小圖用 INTER_AREA 邊緣最漂亮；放大圖用 INTER_CUBIC 最平滑
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized

def main():
    parser = argparse.ArgumentParser(description="批次等比例縮放圖片工具")
    parser.add_argument("--input", default="inputs", help="輸入資料夾或單一檔案路徑")
    parser.add_argument("--output", default="output_resized", help="輸出資料夾")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="設定最長邊的像素大小")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ── 關鍵修改：判斷是檔案還是資料夾 ──
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in sorted(input_path.iterdir()) if f.is_file() and f.suffix.lower() in exts]
    else:
        print(f"[錯誤] 找不到輸入路徑: {input_path}")
        return

    if not files:
        print("[警告] 找不到任何圖片檔案！")
        return

    print(f"=== 開始處理 {len(files)} 個項目 ===")
    for file in files:
        img = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
            
        resized_img = resize_image_keep_aspect_ratio(img, args.size)
        out_path = output_dir / file.name
        is_success, im_buf_arr = cv2.imencode(out_path.suffix, resized_img)
        if is_success:
            im_buf_arr.tofile(str(out_path))
            print(f"處理完成: {file.name} -> {resized_img.shape[1]}x{resized_img.shape[0]}")

if __name__ == "__main__":
    main()
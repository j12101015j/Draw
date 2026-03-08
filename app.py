# app.py
import streamlit as st
import os
import tempfile
import pandas as pd
import zipfile
import py7zr
import rarfile
rarfile.UNRAR_TOOL = r"C:\Program Files\WinRAR\UnRAR.exe"
from io import BytesIO
from ultralytics import YOLO

# 匯入你原本的特徵萃取主程式
from features import extract_features_for_image

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="兒童繪畫特徵分析系統", layout="wide")
st.title("🎨 兒童繪畫特徵與內容分析系統")
st.write("請點擊下方按鈕上傳圖片，或直接上傳包含圖片的 ZIP 壓縮檔，系統將自動進行特徵萃取與 YOLO 物件辨識。")

# --- 初始化網頁的「記憶體」 ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.df_cn = None

# --- 2. 載入 YOLO 模型 ---
@st.cache_resource
def load_model():
    model_path = "models/content.pt" # 確保檔名正確
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning(f"找不到 YOLO 模型檔 ({model_path})，內容辨識功能將無效。")
        return None

yolo_model = load_model()

# --- 3. 建立多檔案上傳區塊 (新增 zip 支援) ---
uploaded_files = st.file_uploader(
    "上傳圖片、ZIP、RAR 或 7Z (支援多選)", 
    type=["jpg", "jpeg", "png", "bmp", "zip", "rar", "7z"],
    accept_multiple_files=True
)

# --- 4. 點擊分析按鈕的處理邏輯 ---
if uploaded_files:
    if st.button("開始大量分析"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        rows_cn = []
        
        temp_out_dir = "temp_web_output"
        os.makedirs(temp_out_dir, exist_ok=True)
        
        # 🌟 建立一個自動清理的暫存資料夾，用來解壓縮和收集所有圖片
        with tempfile.TemporaryDirectory() as process_dir:
            image_paths_to_process = []
            
            status_text.text("正在整理與解壓縮上傳的檔案...")
            
            # 步驟 A: 整理所有上傳的檔案
            for uploaded_file in uploaded_files:
                fname = uploaded_file.name.lower()
                if fname.endswith('.zip'):
                    with zipfile.ZipFile(uploaded_file, 'r') as z:
                        z.extractall(process_dir)
                elif fname.endswith('.7z'):
                    with py7zr.SevenZipFile(uploaded_file, mode='r') as s:
                        s.extractall(process_dir)
                elif fname.endswith('.rar'):
                    with rarfile.RarFile(uploaded_file) as r:
                        r.extractall(process_dir)
                else:
                    # 如果是一般圖片，直接存入 process_dir
                    file_path = os.path.join(process_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
            
            # 步驟 B: 從 process_dir 掃描所有的有效圖片檔
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for root, dirs, files in os.walk(process_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_exts:
                        # 略過 Mac 系統常產生的隱藏垃圾檔 (如 ._image.jpg 或 __MACOSX)
                        if not file.startswith('._') and '__MACOSX' not in root:
                            image_paths_to_process.append(os.path.join(root, file))
            
            total_images = len(image_paths_to_process)
            
            if total_images == 0:
                st.warning("沒有找到任何有效的圖片檔案！請確認 ZIP 檔內容或上傳的圖片格式。")
            else:
                # 步驟 C: 開始排隊分析圖片
                for i, img_path in enumerate(image_paths_to_process):
                    img_name = os.path.basename(img_path)
                    status_text.text(f"正在分析 ({i+1}/{total_images}): {img_name}")
                    
                    try:
                        res = extract_features_for_image(
                            image_path=img_path, 
                            base_out_dir=temp_out_dir, 
                            model=yolo_model
                        )
                        
                        if res:
                            row_cn = {
                                "圖檔名稱": img_name,
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
                            
                    except Exception as e:
                        st.error(f"圖片 {img_name} 處理失敗: {e}")
                    
                    progress_bar.progress((i + 1) / total_images)
                    
                status_text.success(f"🎉 所有圖片 (共 {total_images} 張) 分析完成！")
                
                # 算完之後存入記憶體
                if rows_cn:
                    st.session_state.df_cn = pd.DataFrame(rows_cn)
                    st.session_state.analysis_done = True

# --- 5. 獨立顯示結果與下載按鈕 ---
if st.session_state.analysis_done and st.session_state.df_cn is not None:
    df_cn = st.session_state.df_cn
    
    st.subheader("📊 分析總表預覽")
    st.dataframe(df_cn)
    
    st.write("---")
    col1, col2 = st.columns(2)
    
    # CSV 下載 (強制編碼解決亂碼問題)
    csv = df_cn.to_csv(index=False).encode('utf-8-sig')
    with col1:
        st.download_button(
            label="📥 下載 CSV 總表",
            data=csv,
            file_name="kids_drawing_features.csv",
            mime="text/csv"
        )
    
    # Excel 下載
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_cn.to_excel(writer, index=False, sheet_name='Features')
    excel_data = output.getvalue()
    
    with col2:
        st.download_button(
            label="📥 下載 Excel 總表",
            data=excel_data,
            file_name="kids_drawing_features.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
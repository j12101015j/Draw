# app.py
import streamlit as st
import psutil
import os
import tempfile
import pandas as pd
import zipfile
import py7zr
import rarfile
import gc  # <--- 新增：垃圾回收機制，用來清空記憶體
from io import BytesIO
from ultralytics import YOLO

# 匯入你原本的特徵萃取主程式
from features import extract_features_for_image

# (保留給地端使用的 WinRAR 路徑，雲端因為有 unrar-free 所以不受影響)
rarfile.UNRAR_TOOL = r"C:\Program Files\WinRAR\UnRAR.exe"

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="兒童繪畫特徵分析系統", layout="wide")
st.title("🎨 兒童繪畫特徵與內容分析系統")
st.write("請點擊下方按鈕上傳圖片，或直接上傳包含圖片的 ZIP 壓縮檔，系統將自動進行特徵萃取與 YOLO 物件辨識。")

# --- 初始化網頁的「狀態」 ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.df_cn = None

# --- 2. 記憶體監控器 ---
def show_memory_monitor():
    """在側邊欄顯示記憶體狀態 (精準測量本程式)"""
    # 抓取「當前這個 Python 程式」的進程
    process = psutil.Process(os.getpid())
    
    # 🌟 關鍵修改：只計算這個程式實際佔用的實體記憶體 (RSS)
    used_gb = process.memory_info().rss / (1024 ** 3)
    
    total_gb = 1.0  # Streamlit Cloud 通常分配約 1GB
    percent = (used_gb / total_gb) * 100

    st.sidebar.divider()
    st.sidebar.subheader("🖥️ 雲端資源監測 (精準版)")

    color = "normal"
    if percent > 85:
        color = "inverse"
        st.sidebar.warning("⚠️ 記憶體即將耗盡，請分批處理圖片")

    st.sidebar.metric(label="本程式專屬記憶體", value=f"{used_gb:.2f} GB", delta=f"{percent:.1f}%", delta_color=color)
    st.sidebar.progress(min(percent/100, 1.0))
    st.sidebar.caption(f"註：Streamlit Cloud 免費版上限約 {total_gb} GB")

# 執行監測 (一開網頁就會顯示目前超低的安全水位)
show_memory_monitor() 

# --- 3. 載入模型 (⚠️ 注意：已移除 @st.cache_resource) ---
def load_models():
    models = {}
    
    # 內容模型
    content_path = "models/content.pt"
    if os.path.exists(content_path):
        models['content'] = YOLO(content_path)
    else:
        st.warning(f"找不到內容模型 ({content_path})。")
        models['content'] = None

    # 情緒模型
    mood_path = "models/emotion.pt"
    if os.path.exists(mood_path):
        models['mood'] = YOLO(mood_path)
    else:
        st.warning(f"找不到情緒模型 ({mood_path})。")
        models['mood'] = None

    # 文字模型
    word_path = "models/pure_draw_6_1_best_v8s.pt"
    if os.path.exists(word_path):
        models['word'] = YOLO(word_path)
    else:
        st.warning(f"找不到文字模型 ({word_path})。")
        models['word'] = None

    return models

# --- 4. 建立多檔案上傳區塊 ---
uploaded_files = st.file_uploader(
    "上傳圖片、ZIP、RAR 或 7Z (支援多選)",
    type=["jpg", "jpeg", "png", "bmp", "zip", "rar", "7z"],
    accept_multiple_files=True
)

# --- 5. 點擊分析按鈕的處理邏輯 ---
if uploaded_files:
    if st.button("開始大量分析"):

        progress_bar = st.progress(0)
        status_text = st.empty()
        rows_cn = []

        # 🌟 記憶體優化 A：在這裡才呼叫模型起床上班！
        status_text.text("🔄 正在載入 AI 模型，請稍候 (記憶體將會短暫升高)...")
        yolo_models = load_models()

        temp_out_dir = "temp_web_output"
        os.makedirs(temp_out_dir, exist_ok=True)

        # 建立自動清理的暫存資料夾
        with tempfile.TemporaryDirectory() as process_dir:
            image_paths_to_process = []
            status_text.text("📂 正在整理與解壓縮上傳的檔案...")

            # 步驟 A: 解壓縮檔案
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
                    file_path = os.path.join(process_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

            # 步驟 B: 掃描有效圖片
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for root, dirs, files in os.walk(process_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_exts:
                        if not file.startswith('._') and '__MACOSX' not in root:
                            image_paths_to_process.append(os.path.join(root, file))

            total_images = len(image_paths_to_process)

            if total_images == 0:
                st.warning("沒有找到任何有效的圖片檔案！請確認 ZIP 檔內容或上傳的圖片格式。")
            else:
                # 步驟 C: 排隊分析圖片
                for i, img_path in enumerate(image_paths_to_process):
                    img_name = os.path.basename(img_path)
                    status_text.text(f"🔍 正在分析 ({i+1}/{total_images}): {img_name}")

                    try:
                        res = extract_features_for_image(
                            image_path=img_path,
                            base_out_dir=temp_out_dir,
                            model=yolo_models['content'],
                            mood_model=yolo_models['mood'],
                            word_model=yolo_models['word']
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
                                "情緒": res.row.get("emotion"),
                                "文字": res.row.get("word")
                            }
                            rows_cn.append(row_cn)

                    except Exception as e:
                        st.error(f"圖片 {img_name} 處理失敗: {e}")

                    progress_bar.progress((i + 1) / total_images)

                status_text.success(f"🎉 所有圖片 (共 {total_images} 張) 分析完成！")

                # 存入記憶體供顯示
                if rows_cn:
                    st.session_state.df_cn = pd.DataFrame(rows_cn)
                    st.session_state.analysis_done = True

        # 🌟 記憶體優化 B：分析完畢，強制清空模型與記憶體！
        del yolo_models
        gc.collect()

# --- 6. 獨立顯示結果與下載按鈕 ---
if st.session_state.analysis_done and st.session_state.df_cn is not None:
    df_cn = st.session_state.df_cn

    st.subheader("📊 分析總表預覽")
    st.dataframe(df_cn.astype(str))

    st.write("---")
    col1, col2 = st.columns(2)

    # CSV 下載
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
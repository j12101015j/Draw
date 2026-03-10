# app.py
import streamlit as st
import os
import tempfile
import pandas as pd
import zipfile
import py7zr
import psutil
import torch
import rarfile
import time  # 🌟 新增：匯入時間模組用來計時
rarfile.UNRAR_TOOL = r"C:\Program Files\WinRAR\UnRAR.exe"
from io import BytesIO
from ultralytics import YOLO

# 匯入你原本的特徵萃取主程式
from features import extract_features_for_image

# =======================================================
# ⚙️ 系統控制區 (在這裡手動更改設定)
# =======================================================
FORCE_DEVICE = "cpu"  # 填入 "auto" (自動偵測), "cpu" (強制使用 CPU), 或 "gpu" (強制使用 GPU)
# =======================================================

# --- 決定系統實際運算設備 ---
has_gpu = torch.cuda.is_available()
device_choice = FORCE_DEVICE.lower()

if device_choice == "gpu" and not has_gpu:
    actual_device = "cpu"
elif device_choice == "gpu" and has_gpu:
    actual_device = "cuda:0"
elif device_choice == "cpu":
    actual_device = "cpu"
else: # auto
    actual_device = "cuda:0" if has_gpu else "cpu"

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="兒童繪畫特徵分析系統", layout="wide")
st.title("🎨 兒童繪畫特徵與內容分析系統")
st.write("請點擊下方按鈕上傳圖片，或直接上傳包含圖片的 ZIP 壓縮檔，系統將自動進行特徵萃取與 YOLO 物件辨識。")

# --- 系統資源監控面板 ---
def show_system_status():
    st.sidebar.title("🖥️ 系統即時監控")
    
    # 抓取「當前程式」的記憶體與「整台機器」的記憶體
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    sys_mem = psutil.virtual_memory()
    
    st.sidebar.metric("本程式佔用 RAM", f"{mem_mb:.0f} MB")
    st.sidebar.metric("整台電腦 RAM 使用率", f"{sys_mem.percent} %")
    
    st.sidebar.write("---")
    st.sidebar.subheader("⚙️ 運算核心狀態")
    
    # 動態顯示目前的運算設備
    if actual_device == "cpu":
        if device_choice == "cpu":
            st.sidebar.warning("🐢 純 CPU 運算中\n\n(依據系統控制區強制設定)")
        else:
            st.sidebar.warning("🐢 純 CPU 運算中\n\n(未偵測到 CUDA，速度會較慢)")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 已啟用 GPU 加速\n\n設備: {gpu_name}")

show_system_status()

# --- 狀態初始化 ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.df_cn = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'uploader_key' not in st.session_state:         # 🌟 新增這行：記錄上傳區的 ID
    st.session_state.uploader_key = 0              # 🌟 新增這行：初始值為 0

# 🌟【防呆機制】：當上傳區塊發生改變(新增/刪除檔案)時，強制重置所有狀態！
def on_upload_change():
    st.session_state.is_processing = False
    st.session_state.stop_requested = False
    st.session_state.analysis_done = False
    st.session_state.df_cn = None

# --- 2. 載入 YOLO 模型 ---
@st.cache_resource
def load_models(device):
    models = {}
    content_path = "models/content.pt"
    if os.path.exists(content_path):
        models['content'] = YOLO(content_path)
        models['content'].to(device)
    else:
        models['content'] = None

    mood_path = "models/emotion.pt"
    if os.path.exists(mood_path):
        models['mood'] = YOLO(mood_path)
        models['mood'].to(device)
    else:
        models['mood'] = None

    word_path = "models/pure_draw_6_1_best_v8s.pt"
    if os.path.exists(word_path):
        models['word'] = YOLO(word_path)
        models['word'].to(device)
    else:
        models['word'] = None
        
    return models

# 傳入剛剛決定好的設備
yolo_models = load_models(actual_device)

# --- 3. 建立多檔案上傳區塊 ---
# 🌟 隱藏技：注入 CSS 把上傳列表拉長 (將 max-height 設大一點就可以顯示更多)
st.markdown(
    """
    <style>
    /* 改變上傳檔案列表的最大高度，讓它可以顯示超過 3 個檔案 */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none; /* 隱藏原本佔空間的拖曳提示文字 */
    }
    section[data-testid="stFileUploadDropzone"] {
        padding: 2rem;
    }
    /* 調整列表容器的高度 */
    ul[data-testid="stUploadedFileList"] {
        max-height: 300px; /* 原本大約是 150px，現在加倍，大概可以顯示 5~7 個 */
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "上傳圖片、ZIP、RAR 或 7Z (支援多選)", 
    type=["jpg", "jpeg", "png", "bmp", "zip", "rar", "7z"],
    accept_multiple_files=True,
    on_change=on_upload_change,
    key=f"uploader_{st.session_state.uploader_key}"
)

# 🌟 新增功能：顯示目前已選擇的檔案總數
if uploaded_files:
    file_count = len(uploaded_files)
    st.info(f"📂 目前已選擇 **{file_count}** 個檔案，準備進行分析。")

# --- 4. 操作區塊 (開始、停止與清空) ---
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    start_btn = st.button("🚀 開始大量分析", disabled=st.session_state.is_processing)

with col_btn2:
    if st.session_state.is_processing:
        if st.button("🛑 強制停止分析", type="primary"):
            st.session_state.stop_requested = True
            st.warning("⚠️ 收到停止信號，正在中斷目前任務...")

with col_btn3:
    # 🌟 新增清空按鈕邏輯
    if uploaded_files and not st.session_state.is_processing:
        if st.button("🗑️ 清空所有檔案"):
            st.session_state.uploader_key += 1
            on_upload_change()
            st.rerun()

# --- 5. 點擊分析按鈕的處理邏輯 ---
if uploaded_files and start_btn:
    st.session_state.is_processing = True
    st.session_state.stop_requested = False
    st.session_state.analysis_done = False
    st.rerun() 

if st.session_state.is_processing and not st.session_state.stop_requested:
    
    # 🌟 開始計時
    start_time = time.time()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    rows_cn = []
    
    temp_out_dir = "temp_web_output"
    os.makedirs(temp_out_dir, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as process_dir:
        image_paths_to_process = []
        status_text.text("正在整理與解壓縮上傳的檔案...")
        
        for uploaded_file in uploaded_files:
            fname = uploaded_file.name.lower()
            if fname.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file, 'r') as z: z.extractall(process_dir)
            elif fname.endswith('.7z'):
                with py7zr.SevenZipFile(uploaded_file, mode='r') as s: s.extractall(process_dir)
            elif fname.endswith('.rar'):
                with rarfile.RarFile(uploaded_file) as r: r.extractall(process_dir)
            else:
                file_path = os.path.join(process_dir, uploaded_file.name)
                with open(file_path, "wb") as f: f.write(uploaded_file.getvalue())
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for root, dirs, files in os.walk(process_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts and not file.startswith('._') and '__MACOSX' not in root:
                    image_paths_to_process.append(os.path.join(root, file))
        
        total_images = len(image_paths_to_process)
        
        if total_images == 0:
            st.warning("沒有找到任何有效的圖片檔案！")
            st.session_state.is_processing = False
            st.rerun()
        else:
            for i, img_path in enumerate(image_paths_to_process):
                if st.session_state.stop_requested:
                    status_text.error("🚫 分析已由使用者手動中斷。")
                    break
                
                img_name = os.path.basename(img_path)
                status_text.text(f"正在分析 ({i+1}/{total_images}): {img_name}")
                
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
                
            if not st.session_state.stop_requested:
                # 🌟 結算時間
                end_time = time.time()
                total_seconds = end_time - start_time
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60
                
                # 🌟 將時間顯示在成功的綠色框框裡面
                status_text.success(f"🎉 所有圖片 (共 {total_images} 張) 分析完成！總共耗時: {minutes} 分 {seconds:.2f} 秒")
            
            if rows_cn:
                st.session_state.df_cn = pd.DataFrame(rows_cn)
                st.session_state.analysis_done = True
                
            st.session_state.is_processing = False
            # 🌟 注意：拿掉結尾的 st.rerun()，讓使用者能看清楚成功與耗時的訊息！
            # st.rerun()

# --- 6. 獨立顯示結果與下載按鈕 ---
if st.session_state.analysis_done and st.session_state.df_cn is not None:
    df_cn = st.session_state.df_cn
    st.subheader("📊 分析總表預覽")
    st.dataframe(df_cn.astype(str))
    st.write("---")
    col1, col2 = st.columns(2)
    csv = df_cn.to_csv(index=False).encode('utf-8-sig')
    with col1:
        st.download_button(label="📥 下載 CSV 總表", data=csv, file_name="kids_drawing_features.csv", mime="text/csv")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_cn.to_excel(writer, index=False, sheet_name='Features')
    with col2:
        st.download_button(label="📥 下載 Excel 總表", data=output.getvalue(), file_name="kids_drawing_features.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
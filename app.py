# app.py
import streamlit as st
import os
import tempfile
import pandas as pd
from io import BytesIO
from ultralytics import YOLO

# 匯入你原本的特徵萃取主程式
from features import extract_features_for_image

# --- 1. 網頁基本設定 ---
st.set_page_config(page_title="兒童繪畫特徵分析系統", layout="wide")
st.title("🎨 兒童繪畫特徵與內容分析系統")
st.write("請點擊下方按鈕上傳圖片（可一次全選多張），系統將自動進行特徵萃取與 YOLO 物件辨識。")

# --- 2. 載入 YOLO 模型 ---
# @st.cache_resource 確保模型只會載入一次，不會每次按按鈕都重新讀取
@st.cache_resource
def load_model():
    # 改用相對路徑！這樣上雲端才找得到檔案
    model_path = "models/best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning("找不到 YOLO 模型檔 (models/best.pt)，內容辨識功能將無效。")
        return None

yolo_model = load_model()

# --- 3. 建立多檔案上傳區塊 ---
uploaded_files = st.file_uploader(
    "上傳圖片 (支援多選)", 
    type=["jpg", "jpeg", "png", "bmp"], 
    accept_multiple_files=True # 開啟多選功能！
)

# 如果使用者有上傳檔案
if uploaded_files:
    if st.button("開始大量分析"):
        
        # 建立進度條
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 準備一個列表來裝所有圖片的中文分析結果
        rows_cn = []
        
        # 建立一個暫存的輸出資料夾，給 features.py 產生中間圖片用 (雖然網頁不顯示，但程式需要)
        temp_out_dir = "temp_web_output"
        os.makedirs(temp_out_dir, exist_ok=True)
        
        # --- 4. 開始處理每一張圖片 ---
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"正在分析 ({i+1}/{len(uploaded_files)}): {uploaded_file.name}")
            
            # 將網頁上傳的檔案存成實體的暫存檔，讓 OpenCV 可以讀取
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # 呼叫你原本的核心程式！
                res = extract_features_for_image(
                    image_path=tmp_path, 
                    base_out_dir=temp_out_dir, 
                    model=yolo_model
                )
                
                if res:
                    # 照抄 identify.py 的中文對應邏輯
                    row_cn = {
                        "圖檔名稱": uploaded_file.name, # 使用原本上傳的檔名
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
                st.error(f"圖片 {uploaded_file.name} 處理失敗: {e}")
                
            finally:
                # 處理完這張圖片就刪除暫存檔，節省空間
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            # 更新進度條
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.success("🎉 所有圖片分析完成！")
        
        # --- 5. 顯示結果與下載按鈕 ---
        if rows_cn:
            df_cn = pd.DataFrame(rows_cn)
            
            st.subheader("📊 分析總表預覽")
            # Streamlit 的 dataframe 會自動調整一個漂亮且可互動的表格
            st.dataframe(df_cn)
            
            st.write("---")
            col1, col2 = st.columns(2)
            
            # 下載 CSV 按鈕
            csv = df_cn.to_csv(index=False, encoding="utf-8-sig")
            with col1:
                st.download_button(
                    label="📥 下載 CSV 總表",
                    data=csv,
                    file_name="kids_drawing_features.csv",
                    mime="text/csv"
                )
            
            # 下載 Excel 按鈕 (在記憶體中生成 Excel)
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
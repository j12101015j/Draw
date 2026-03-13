# api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import gc
from ultralytics import YOLO

# 匯入你的核心大腦！
from features import extract_features_for_image

# --- 1. 啟動與設定 API ---
api_description = """
本 API 負責接收兒童畫作，並透過 AI 模型自動萃取 20 項關鍵特徵數據。

### 📊 完整數據字典 (Data Dictionary)
| JSON 欄位名稱 | 對應需求維度 / 描述 |
| :--- | :--- |
| **image** | 原始上傳影像檔名 |
| **paper_orientation** | 紙張放置方向 (1: 縱向, 2: 橫向) |
| **curvature** | 畫面弧度分數(1~10分) | 
| **line_smoothness** | 線條流暢度分數(1~10分) |
| **line_thickness** | 線條粗細程度分數(1~10分) |
| **colors_rgb** | 畫面主要顏色的 RGB 十六進位值列表 |
| **colors_name** | 主要顏色之中文與英文名稱對照 |
| **color_count** | 畫作中使用的顏色總數統計 |
| **shadow_area_ratio** | 陰影區域佔整張紙張之"比例" |
| **shadow_region_count** | 偵測到的陰影區塊總"個數" |
| **drawing_region_main** | 畫面最主要分布的九宮格"區域範圍" (如 C2, B1) |
| **drawing_region_covered** | 繪圖線條所涵蓋的所有"區域" |
| **drawing_area_ratio** | 所有繪圖內容佔整張紙張之總"比例" |
| **stroke_depth** | 繪畫力度(調整中) |
| **content** | 偵測到的物件類別與數量 |
| **content_size_all** | 各類物件佔"辨識到的總繪圖物件"的比例 |
| **content_size_paper** | 各類物件佔"全紙張面積"的比例 |
| **dynamic** | 畫面呈現之事物動態性分數(1~10分) |
| **emotion** | 綜合情緒指標辨識分數(1~10分) |
| **word** | 影像中辨識出的具體"數字"內容 (若無則顯示 None) |

---

### 🚀 串接範例 (純 Binary 傳輸模式)
本 API 採用 `multipart/form-data` 接收檔案。

**【方式一：使用 cURL 指令】**
```bash
curl -X 'POST' 'https://naples-screen-scheduled-corner.trycloudflare.com/docs#/default/analyze_drawing' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: multipart/form-data' \\
  -F 'file=@image.jpg;type=image/jpeg'

【方式二：使用 Python requests 測試】
import requests
url = "https://naples-screen-scheduled-corner.trycloudflare.com/docs#/default/analyze_drawing"
image_path = r"image.jpg"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
"""

app = FastAPI(title="🏥 兒童繪畫特徵分析 API", description=api_description, version="1.0.0")

# 加入 CORS 防護網解鎖，避免前端網頁連線失敗
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 預先載入模型 (移到最外面，大幅加快運算速度，避免斷線！) ---
print("⏳ 正在預先載入 YOLO 模型，請稍候...")
MODELS = {}

if os.path.exists("models/content.pt"):
    MODELS['content'] = YOLO("models/content.pt")
else:
    MODELS['content'] = None

if os.path.exists("models/emotion.pt"):
    MODELS['mood'] = YOLO("models/emotion.pt")
else:
    MODELS['mood'] = None

if os.path.exists("models/pure_draw_6_1_best_v8s.pt"):
    MODELS['word'] = YOLO("models/pure_draw_6_1_best_v8s.pt")
else:
    MODELS['word'] = None

print("✅ 模型載入完成！API 伺服器已準備就緒。")


# --- 3. 唯一的投幣口：接收二進位圖片 ---
@app.get("/")
async def root():
    return {"message": "歡迎來到兒童繪畫辨識 API 大門，請前往 /docs 查看說明"}
@app.post(
    "/analyze_drawing",
    summary="上傳畫作並取得 20 項特徵 JSON"
)
async def analyze_drawing(file: UploadFile = File(...)):
    
    # 建立暫存區來放醫院傳來的二進位圖片
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_image_path = os.path.join(temp_dir, file.filename)
        
        # 將 Binary 資料寫入成實體圖片檔
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await file.read())
        
        temp_out_dir = os.path.join(temp_dir, "output")
        os.makedirs(temp_out_dir, exist_ok=True)
        
        # 呼叫核心大腦算分數 (直接使用已經載入好的 MODELS)
        try:
            res = extract_features_for_image(
                image_path=temp_image_path,
                base_out_dir=temp_out_dir,
                model=MODELS['content'],
                mood_model=MODELS['mood'],
                word_model=MODELS['word']
            )
            
            # 把所有特徵轉成 JSON 字典格式
            if res:
                result_data = {
                    "image": file.filename,                                  
                    "paper_orientation": res.row.get("paper_orientation"),   
                    "curvature": res.row.get("curvature"),                   
                    "line_smoothness": res.row.get("line_smoothness"),       
                    "line_thickness": res.row.get("line_thickness"),         
                    "colors_rgb": res.row.get("colors_rgb"),                 
                    "colors_name": res.row.get("colors_name"),               
                    "color_count": res.row.get("color_count"),               
                    "shadow_area_ratio": res.row.get("shadow_area_ratio"),   
                    "shadow_region_count": res.row.get("shadow_region_count"),
                    "drawing_region_main": res.row.get("drawing_region_main"),
                    "drawing_region_covered": res.row.get("drawing_region_covered"), 
                    "drawing_area_ratio": res.row.get("drawing_area_ratio"), 
                    "stroke_depth": res.row.get("stroke_depth"),             
                    "content": res.row.get("content"),                       
                    "content_size_all": res.row.get("content_size_all"),     
                    "content_size_paper": res.row.get("content_size_paper"), 
                    "dynamic": res.row.get("dynamic"),                       
                    "emotion": res.row.get("emotion"),                       
                    "word": res.row.get("word")                              
                }
            else:
                result_data = {"error": "分析失敗，未回傳結果"}
                
        except Exception as e:
            result_data = {"error": str(e)}
        
        # 定期清垃圾，避免記憶體爆掉
        gc.collect()
        
        # 原路退回給醫院！
        return {"status": "success", "data": result_data}
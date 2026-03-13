# 兒童繪畫特徵分析系統1.0(測試中) - API 測試環境包

本專案提供兒童繪畫特徵（包含傳統影像特徵與 YOLO 深度學習物件/情緒辨識）的 API 服務。

## 📁 檔案結構說明

請確保您的資料夾結構如下，特別是 `models` 資料夾與其中的權重檔不可遺漏：

├── api.py		# API 主程式 (FastAPI 進入點)
├── features.py 	# 特徵萃取主程式 (整合各項分析)
├── bina_sk.py 	# 影像前處理 (二值化與骨架化)
├── color.py 	# 色彩分析模組
├── angle.py 	# 角度與弧度分析模組
├── thickness.py 	# 粗細度分析模組
├── fluency.py 	# 流暢度分析模組
├── requirements.txt # 環境套件依賴清單
├── models/ 		# 深度學習模型權重資料夾
│   ├── content.pt
│   ├── emotion.pt
│   └── pure_draw_6_1_best_v8s.pt
│
│
└── 開發者測試工具 (非必要)
    └── test_api.py      # 本地端 API 批次測試腳本 (支援資料夾與結果輸出)


## 🛠️ 環境建置 (Environment Setup)

建議使用 Python 3.8 或以上版本。請在終端機 (Terminal) 或命令提示字元中，切換到本專案目錄，並執行以下指令安裝所需套件：
pip install -r requirements.txt

# 啟動API伺服器
uvicorn api:app --reload

伺服器預設將運行在：http://127.0.0.1:8000

啟動成功後，您可以直接在瀏覽器開啟 Swagger UI 互動式說明文件 進行測試：
👉 http://127.0.0.1:8000/docs

在 Docs 介面中，您可以直接上傳圖片並測試 API 的 JSON 回傳結果，不需額外撰寫前端程式即可驗證。
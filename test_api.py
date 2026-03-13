# test_api.py
import os
import requests
import json
import time

API_URL = "http://127.0.0.1:8000/analyze_drawing"

# 🌟 在這裡填寫你的路徑 (可以是單張圖片，也可以是整個資料夾)
TARGET_PATH = r"D:\XTX\Yehlab\Draw\圖片\sample3_re" 

# 準備一個大箱子，用來收集所有圖片的回傳結果
ALL_RESULTS = {}

def send_to_api(image_path):
    """負責把圖片射給 API 並處理結果的函式"""
    img_name = os.path.basename(image_path)
    print(f"\n發送中 ➔ [{img_name}] ...")
    
    with open(image_path, "rb") as image_file:
        files = {"file": (img_name, image_file, "image/jpeg")}
        start_time = time.time()
        
        try:
            response = requests.post(API_URL, files=files)
            cost_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 成功！(耗時 {cost_time:.2f} 秒)")
                
                # 1. 偷偷把完整的 JSON 存進大箱子裡，最後會生出實體檔案
                ALL_RESULTS[img_name] = result
                
                # 2. 終端機先印出幾項重點，讓你知道有算出來
                data = result.get("data", {})
                # ==========================================
                # 🌟 完整 JSON 顯示開關 (展示給醫院看你真的有吐 JSON！)
                # 如果要打開，請把下面這兩行前面的 `#` 刪掉：
                # ==========================================
                print("   👇 完整 JSON 資料如下：")
                print(json.dumps(result, indent=4, ensure_ascii=False))
                
            else:
                print(f"❌ API 發生錯誤，狀態碼: {response.status_code}")
                print(response.text)
        except requests.exceptions.ConnectionError:
            print("❌ 連線失敗！請確認你的 API 伺服器 (uvicorn) 有沒有啟動？")

# ==========================================
# 🧠 智慧判定邏輯區 (自動判斷單張圖片 or 資料夾)
# ==========================================
if not os.path.exists(TARGET_PATH):
    print(f"⚠️ 找不到路徑，請檢查路徑是否正確：\n{TARGET_PATH}")

elif os.path.isfile(TARGET_PATH):
    print("🏥 醫院端模式：[單張圖片測試]")
    send_to_api(TARGET_PATH)

elif os.path.isdir(TARGET_PATH):
    print("🏥 醫院端模式：[資料夾批次測試]")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in os.listdir(TARGET_PATH) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not images:
        print("⚠️ 這個資料夾裡面沒有找到任何圖片檔喔！")
    else:
        print(f"📂 共找到 {len(images)} 張圖片，準備開始發送...")
        for img_name in images:
            full_path = os.path.join(TARGET_PATH, img_name)
            send_to_api(full_path)
import cv2
import numpy as np
import math

def draw_wavy(img, y, thickness, x0, x1, amp=10, period=90, color=(0, 0, 0)):
    pts = []
    for x in range(x0, x1):
        yy = int(y + amp * math.sin(2 * math.pi * (x - x0) / period))
        pts.append((x, yy))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, lineType=cv2.LINE_AA)

def main():
    W, H = 1400, 900
    img = np.full((H, W, 3), 255, np.uint8)

    # ✅ 依需求表：1~3 很細(鉛筆)、8~10 正常彩色筆、4~7 超粗(>正常彩色筆)
    # 你可以視覺上再調整這些 px，但分段邏輯就是這樣
    level_to_px = {
        1: 1, 2: 2, 3: 3,        # pencil thin
        4: 14, 5: 16, 6: 18, 7: 20,  # extra thick (thicker than normal marker)
        8: 6, 9: 7, 10: 8        # normal marker
    }

    cv2.putText(img, "Thickness Levels Demo (rule: 1-3 thin/pencil, 8-10 normal marker, 4-7 extra thick)",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, "Each row: left=straight, right=wavy", (40, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,60), 2, cv2.LINE_AA)

    y0, gap = 150, 65
    for lvl in range(1, 11):
        t = int(level_to_px[lvl])
        y = y0 + (lvl - 1) * gap

        # 分段提示（照你的規則）
        if lvl <= 3:
            tag = "THIN (pencil)"
        elif 4 <= lvl <= 7:
            tag = "EXTRA THICK (> marker)"
        else:
            tag = "NORMAL MARKER"

        cv2.putText(img, f"Level {lvl}  px={t}  [{tag}]", (40, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)

        # 參考中線
        cv2.line(img, (40, y), (1360, y), (230,230,230), 1, cv2.LINE_AA)

        # 直線 + 波浪線
        cv2.line(img, (40, y), (650, y), (0,0,0), t, cv2.LINE_AA)
        draw_wavy(img, y, t, x0=760, x1=1360, amp=10, period=90, color=(0,0,0))

    out = "thickness_levels_demo_v2.png"
    cv2.imwrite(out, img)
    print(f"[OK] Saved: {out}")

if __name__ == "__main__":
    main()

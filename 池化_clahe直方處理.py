import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # 進度條庫

def pooled_clahe(image, pool_size=3, clip_limit=5, grid_size=(16, 16)):
    """
    平均池化 + CLAHE 影像增強
    :param image: 輸入影像（BGR）
    :param pool_size: 平均池化的範圍 (pool_size x pool_size)
    :param clip_limit: CLAHE 的對比增強限制
    :param grid_size: CLAHE 的網格大小
    :return: 增強後的影像
    """
    # 轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 進行平均池化（使用模糊模擬）
    pooled_gray = cv2.blur(gray, (pool_size, pool_size))

    # 建立 CLAHE 物件
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # 套用 CLAHE 增強
    enhanced_image = clahe.apply(pooled_gray)

    return enhanced_image

# 讀取影像
# 設定來源資料夾和輸出資料夾
input_folder = r"C:\Users\benker\Downloads\aoi\train_images"   # 替換成你的來源資料夾名稱
output_folder =r"C:\Users\benker\Downloads\aoi\ttain_images_trans" # 替換成你的輸出資料夾名稱

# 如果輸出資料夾不存在，則建立它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 獲取所有影像檔案名稱
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

# 讀取資料夾內所有影像並處理（帶進度條）
for filename in tqdm(image_files, desc="處理進度", unit="張"):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 讀取影像
    image = cv2.imread(input_path)
    if image is None:
        print(f"無法讀取 {filename}，跳過...")
        continue

    # 進行梯度加權直方圖轉換
    transformed_image = pooled_clahe(image)

    # 儲存轉換後的影像
    cv2.imwrite(output_path, transformed_image)

print("✅ 所有影像處理完成！")

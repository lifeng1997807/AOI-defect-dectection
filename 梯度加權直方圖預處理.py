# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 03:52:59 2025

@author: benker
"""

import cv2
import numpy as np
import os
from tqdm import tqdm  # 進度條庫

def gradient_weighted_histogram(image):
    # 轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 計算 X 和 Y 方向的 Sobel 梯度
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X 梯度
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y 梯度

    # 計算梯度強度
    magnitude = np.sqrt(Gx**2 + Gy**2)

    # 初始化梯度加權直方圖
    hist = np.zeros(256)

    # 計算每個灰階值的梯度加權直方圖
    for i in range(256):
        hist[i] = np.sum(magnitude[gray == i])

    # 歸一化直方圖，使其範圍在 0～255 之間
    hist = hist / np.max(hist) * 255

    # 建立 LUT (查找表) 進行影像轉換
    lut = np.uint8(hist)

    # 轉換影像
    transformed_image = cv2.LUT(gray, lut)

    return transformed_image

# 設定來源資料夾和輸出資料夾
input_folder = r"C:\Users\benker\Downloads\aoi\test_images"   # 替換成你的來源資料夾名稱
output_folder =r"C:\Users\benker\Downloads\aoi\test_images_trans" # 替換成你的輸出資料夾名稱

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
    transformed_image = gradient_weighted_histogram(image)

    # 儲存轉換後的影像
    cv2.imwrite(output_path, transformed_image)

print("✅ 所有影像處理完成！")

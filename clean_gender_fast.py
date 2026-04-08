# File: clean_gender_fast.py

import os
import shutil
import cv2
import numpy as np
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
SOURCE_DIR = r'D:\tuấn\fashion_style_project\data\imaterialist\male_subset'
FEMALE_DIR = r'D:\tuấn\fashion_style_project\data\imaterialist\detected_female'

if not os.path.exists(FEMALE_DIR):
    os.makedirs(FEMALE_DIR)

# --- 2. HÀM XỬ LÝ TỪNG ẢNH ---
def process_single_image(img_name):
    img_path = os.path.join(SOURCE_DIR, img_name)
    try:
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return "Error"
        img_small = cv2.resize(img, (224, 224))
        results = DeepFace.analyze(img_small, actions=['gender'], enforce_detection=False, silent=True)
        gender = results[0]['dominant_gender']
        
        if gender == "Woman":
            shutil.move(img_path, os.path.join(FEMALE_DIR, img_name))
            return "Female"
        return "Male"
    except:
        return "Error"

# --- 3. CHƯƠNG TRÌNH CHÍNH CHẠY ĐA LUỒNG ---
def clean_data_fast():
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total = len(image_files)
    
    print(f" Bắt đầu lọc NHANH {total} ảnh bằng đa luồng...")
    print(f" Thư mục gốc: {SOURCE_DIR}")
    print(f" Thư mục chứa ảnh nữ: {FEMALE_DIR}")
    print("-" * 50)

    count_removed = 0
    count_checked = 0
    start_time = time.time()

    # Sử dụng ThreadPoolExecutor để chạy song song (mặc định 4 luồng)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_img = {executor.submit(process_single_image, img): img for img in image_files}
        for future in as_completed(future_to_img):
            result = future.result()
            count_checked += 1
            if result == "Female":
                count_removed += 1
            if count_checked % 5 == 0 or count_checked == total:
                percent = (count_checked / total) * 100
                elapsed = time.time() - start_time
                speed = count_checked / elapsed if elapsed > 0 else 0
                print(f"\r⏳ Tiến độ: {count_checked}/{total} ({percent:.1f}%) | Đã loại: {count_removed} nữ | Tốc độ: {speed:.1f} ảnh/giây", end="")

    print(f"\n\n✨ HOÀN TẤT!")
    print(f" Tổng số ảnh Nam còn lại: {total - count_removed}")
    print(f" Thời gian thực hiện: {elapsed/60:.2f} phút")

if __name__ == "__main__":
    clean_data_fast()
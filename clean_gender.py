import os
import shutil
import cv2
import numpy as np
from deepface import DeepFace

# --- CẤU HÌNH ---
SOURCE_DIR = r'D:\tuấn\fashion_style_project\data\imaterialist\male_subset'
FEMALE_DIR = r'D:\tuấn\fashion_style_project\data\imaterialist\detected_female'

if not os.path.exists(FEMALE_DIR):
    os.makedirs(FEMALE_DIR)

def clean_data():
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f" Đang xử lý {len(image_files)} ảnh (Đã hỗ trợ đường dẫn tiếng Việt)...")

    count_removed = 0
    count_checked = 0

    for img_name in image_files:
        img_path = os.path.join(SOURCE_DIR, img_name)
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                print(f"⚠️ Không thể đọc ảnh: {img_name}")
                continue
            results = DeepFace.analyze(img, actions=['gender'], enforce_detection=False, silent=True)
            
            gender = results[0]['dominant_gender']
            
            if gender == "Woman":
                shutil.move(img_path, os.path.join(FEMALE_DIR, img_name))
                count_removed += 1
            
            count_checked += 1
            if count_checked % 100 == 0:
                print(f" Đã kiểm tra {count_checked} ảnh. Đã loại bỏ {count_removed} ảnh nữ.")

        except Exception as e:
            continue

    print(f"\n HOÀN TẤT!")
    print(f" Tổng số ảnh đã kiểm tra: {count_checked}")
    print(f" Số ảnh nữ đã loại bỏ: {count_removed}")

if __name__ == "__main__":
    clean_data()
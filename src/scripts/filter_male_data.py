import pandas as pd
import json
import os
import shutil

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = r'D:\tuấn\fashion_style_project'
DATA_DIR = os.path.join(BASE_DIR, r'data\imaterialist')
CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
JSON_PATH = os.path.join(DATA_DIR, 'label_descriptions.json')
IMAGE_DIR = os.path.join(DATA_DIR, 'train') 
OUTPUT_DIR = os.path.join(DATA_DIR, 'male_subset')

def filter_male_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Danh sách ID đồ Nam chuẩn từ kết quả in ra của bạn
    male_ids_to_filter = [0, 1, 2, 4, 5, 6, 7, 9, 16, 18, 19]
    print(f" Đang lọc đồ nam với các ID: {male_ids_to_filter}")

    # 2. Đọc CSV
    df = pd.read_csv(CSV_PATH)
    target_col = 'ClassId' if 'ClassId' in df.columns else 'AttributesIds'

    def is_male_item(val):
        if pd.isna(val): return False
        parts = str(val).replace('_', ',').split(',')
        return any(int(p.strip()) in male_ids_to_filter for p in parts if p.strip().isdigit())

    male_df = df[df[target_col].apply(is_male_item)]
    male_image_ids = male_df['ImageId'].unique()
    print(f" Tìm thấy {len(male_image_ids)} ảnh khớp trong CSV.")

    # 3. Copy ảnh (Sửa lỗi tìm file)
    print(" Đang kiểm tra và copy ảnh...")
    count = 0
    for img_id in male_image_ids:
        file_names = [f"{img_id}", f"{img_id}.jpg", f"{img_id}.png"]
        found = False
        
        for name in file_names:
            src = os.path.join(IMAGE_DIR, name)
            if os.path.exists(src):
                dst = os.path.join(OUTPUT_DIR, name if '.jpg' in name else f"{name}.jpg")
                shutil.copy(src, dst)
                found = True
                count += 1
                break
        
        if count % 500 == 0 and count > 0:
            print(f"  > Đã lưu {count} ảnh...")

    if count == 0:
        print(f" VẪN LỖI: Không tìm thấy file ảnh nào trong thư mục: {IMAGE_DIR}")
        print(f" Hãy kiểm tra xem trong folder '{IMAGE_DIR}' có chứa các file ảnh trực tiếp không hay còn folder con nào khác.")
    else:
        print(f" THÀNH CÔNG! Đã lưu {count} ảnh Nam vào: {OUTPUT_DIR}")

if __name__ == "__main__":
    filter_male_images()
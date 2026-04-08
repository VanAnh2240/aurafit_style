# File: preprocess.py

import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import config

def main():
    print("Bắt đầu tiền xử lý: Cắt ảnh dựa trên list_bbox.txt")
    
    # 1. Tải BBox
    bbox_df = pd.read_csv(config.BBOX_FILE, delim_whitespace=True, skiprows=2, header=None)
    bbox_df.columns = ['image_name', 'x1', 'y1', 'x2', 'y2']
    bbox_map = dict(zip(bbox_df.image_name, bbox_df[['x1', 'y1', 'x2', 'y2']].values))
    
    print(f"Đã tải {len(bbox_map)} bounding boxes.")
    
    # 2. Lấy danh sách TOÀN BỘ ảnh
    partition_df = pd.read_csv(config.PARTITION_FILE, delim_whitespace=True, skiprows=2, header=None)
    partition_df.columns = ['image_name', 'evaluation_status']
    all_images = partition_df['image_name'].tolist()

    print(f"Sẵn sàng xử lý {len(all_images)} ảnh...")
    
    processed_count = 0
    error_count = 0
    
    # 3. Vòng lặp cắt ảnh
    for img_name in tqdm(all_images, desc="Đang cắt ảnh"):
        img_path = config.IMG_DIR / img_name
        output_path = config.CROPPED_IMG_DIR / img_name
        
        try:

            bbox = bbox_map.get(img_name)
            if bbox is None:
                raise FileNotFoundError(f"Không tìm thấy BBox cho {img_name}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(img_path) as img:

                cropped_img = img.crop(bbox)
                cropped_img.save(output_path, format="JPEG")
                
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            
    print("\n--- Tiền xử lý hoàn tất! ---")
    print(f"Đã xử lý thành công: {processed_count} ảnh")
    print(f"Lỗi (không tìm thấy ảnh/bbox): {error_count} ảnh")
    print(f"Dữ liệu đã cắt được lưu tại: {config.CROPPED_IMG_DIR}")

if __name__ == "__main__":
    main()
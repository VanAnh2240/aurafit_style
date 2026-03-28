import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import random

# Import từ các file có sẵn
import config 
from src.dataset import FashionStyleDataset
from src.metrics import calculate_metrics

# Import các hàm chúng ta cần từ train.py (để không lặp code)
from train import get_model, train_one_epoch, evaluate

# Import sklearn để chia K-Fold
from sklearn.model_selection import KFold

def load_full_train_val_data():
    """Tải toàn bộ danh sách ảnh 'train' và 'val' để chia K-Fold."""
    
    print("Đang tải danh sách ảnh 'train' và 'val'...")
    partition_df = pd.read_csv(config.PARTITION_FILE, delim_whitespace=True, skiprows=2, header=None)
    partition_df.columns = ['image_name', 'evaluation_status']
    
    train_images = partition_df[partition_df['evaluation_status'] == 'train']['image_name'].tolist()
    val_images = partition_df[partition_df['evaluation_status'] == 'val']['image_name'].tolist()
    
    full_image_list = train_images + val_images
    
    # --- ÁP DỤNG GIẢM TẢI 50% ---
    frac_to_use = 0.5 # Bạn có thể sửa % ở đây
    num_samples = int(len(full_image_list) * frac_to_use)
    full_image_list = random.Random(42).sample(full_image_list, num_samples)
    
    print(f"Tổng số ảnh (Train+Val) sẽ dùng cho CV: {len(full_image_list)} ({frac_to_use*100}%)")
    return full_image_list

def main(args):
    """Hàm main điều khiển K-Fold CV và lưu trữ nhiều độ đo."""
    
    device = config.DEVICE
    print(f"Sử dụng thiết bị: {device}")
    
    # --- 1. Tải và Chia Dữ liệu ---
    all_images = load_full_train_val_data()
    
    N_SPLITS = 5 # Chia 5-fold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # 1. KHAI BÁO CÁC LIST ĐỂ LƯU KẾT QUẢ TỐT NHẤT CỦA MỖI FOLD
    fold_results = []      
    p3_results = []         
    ndcg5_results = []      
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(all_images)):
        print(f"\n=======================================================")
        print(f" BẮT ĐẦU FOLD {fold + 1}/{N_SPLITS} - Hệ thống: {args.system} ")
        print(f"=======================================================")
        
        # Tạo danh sách ảnh cho fold này
        train_image_list = [all_images[i] for i in train_indices]
        val_image_list = [all_images[i] for i in val_indices]
        
        # --- 2. Chuẩn bị Dataset & DataLoader ---
        train_dataset = FashionStyleDataset(
            mode='train', 
            use_cropped_data=args.use_cropped_data, 
            image_list_override=train_image_list
        )
        val_dataset = FashionStyleDataset(
            mode='val', 
            use_cropped_data=args.use_cropped_data, 
            image_list_override=val_image_list
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True
        )
        
        # --- 3. Khởi tạo Model (cho mỗi fold) ---
        num_styles = train_dataset.num_styles
        num_categories = config.NUM_CATEGORIES
        model = get_model(args.system, num_styles, num_categories).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # --- 4. Vòng lặp Huấn luyện (cho fold này) ---
        best_val_map = 0.0
        best_p3 = 0.0      
        best_ndcg5 = 0.0  
        
        NUM_CV_EPOCHS = 10 
        
        for epoch in range(NUM_CV_EPOCHS):
            print(f"\n--- Fold {fold+1}/{N_SPLITS} -- Epoch {epoch+1}/{NUM_CV_EPOCHS} ---")
            
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.system)
            print(f"Fold {fold+1} Epoch {epoch+1} Train Loss: {train_loss:.4f}")
            
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.system)
            
            print(f"Fold {fold+1} Epoch {epoch+1} Val Loss: {val_loss:.4f}")
            
            # IN RA 3 ĐỘ ĐO CHÍNH XÁC
            print(f"  mAP:     {val_metrics['mAP']:.4f}")
            print(f"  P@3:     {val_metrics['P@3']:.4f}")    
            print(f"  NDCG@5:  {val_metrics['NDCG@5']:.4f}") # Sửa 'ndcg@5' thành 'NDCG@5'

            current_map = val_metrics['mAP']
            if current_map > best_val_map:
                best_val_map = current_map
                # 2. LƯU CẢ 3 ĐỘ ĐO KHI ĐẠT mAP TỐT NHẤT
                best_p3 = val_metrics['P@3']
                best_ndcg5 = val_metrics['NDCG@5']
                print(f"==> Fold {fold+1} mAP cải thiện: {best_val_map:.4f}. Đã lưu P@3 và NDCG@5 tốt nhất.")
        
        # LƯU KẾT QUẢ TỐT NHẤT CỦA FOLD VÀO LIST TỔNG
        fold_results.append(best_val_map)
        p3_results.append(best_p3)
        ndcg5_results.append(best_ndcg5)
        
        print(f"--- FOLD {fold+1} HOÀN TẤT --- mAP tốt nhất: {best_val_map:.4f} ---")

    # --- 5. Báo cáo kết quả cuối cùng ---
    print("\n\n=======================================================")
    print(f" KẾT QUẢ CROSS-VALIDATION (5-Fold) cho: {args.system} ")
    print("=======================================================")
    
    # In kết quả mAP của từng Fold
    for i, map_score in enumerate(fold_results):
        print(f"  Fold {i+1} Best mAP: {map_score:.4f}")
        
    # Tính toán và in ra kết quả trung bình cho CẢ 3 ĐỘ ĐO
    mean_map = np.mean(fold_results)
    std_map = np.std(fold_results)
    
    mean_p3 = np.mean(p3_results)
    std_p3 = np.std(p3_results)
    
    mean_ndcg5 = np.mean(ndcg5_results)
    std_ndcg5 = np.std(ndcg5_results)

    print("\n--- TỔNG KẾT ---")
    print(f"  mAP Trung bình:    {mean_map:.4f} (Độ lệch chuẩn: {std_map:.4f})")
    print(f"  P@3 Trung bình:    {mean_p3:.4f} (Độ lệch chuẩn: {std_p3:.4f})")
    print(f"  NDCG@5 Trung bình: {mean_ndcg5:.4f} (Độ lệch chuẩn: {std_ndcg5:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện K-Fold Cross-Validation")
    
    parser.add_argument(
        '--system', 
        type=str, 
        required=True, 
        choices=['system_1', 'system_2', 'system_3'],
        help='Tên hệ thống để chạy CV (vd: "system_1")'
    )
    
    parser.add_argument(
        '--use_cropped_data',
        action='store_true', 
        help='(Dùng cho Hệ thống 4) Chạy với dữ liệu đã cắt.'
    )
    
    args = parser.parse_args()
    main(args)
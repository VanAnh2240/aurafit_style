# File: evaluate.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import config

from train import evaluate, get_model
from src.dataset import FashionStyleDataset

def run_evaluation(system_name, checkpoint_path, use_cropped):
    """
    Hàm chính để chạy đánh giá cho MỘT hệ thống.
    """
    device = config.DEVICE
    print(f"\n--- Đang đánh giá: {system_name} ---")
    if use_cropped:
        print("Sử dụng dữ liệu ĐÃ CẮT (CROPPED)")
        
    # 1. Tải tập TEST
    test_dataset = FashionStyleDataset(mode='test', use_cropped_data=use_cropped)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    num_styles = test_dataset.num_styles
    num_categories = config.NUM_CATEGORIES
    
    # 2. Tải Model
    model = get_model(system_name, num_styles, num_categories).to(device)
    
    # 3. Tải trọng số (weights) đã lưu
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Đã tải checkpoint từ: {checkpoint_path}")
    except Exception as e:
        print(f"LỖI: Không thể tải checkpoint {checkpoint_path}. Lỗi: {e}")
        return None

    # 4. Chạy đánh giá
    dummy_criterion = torch.nn.BCEWithLogitsLoss() 
    
    test_loss, test_metrics = evaluate(
        model, test_loader, dummy_criterion, device, system_name
    )
    
    print(f"Kết quả trên tập TEST cho {system_name}:")
    print(f"  mAP:    {test_metrics['mAP']:.4f}")
    print(f"  P@3:    {test_metrics['P@3']:.4f}")
    print(f"  NDCG@5: {test_metrics['NDCG@5']:.4f}")
    
    return test_metrics

def main():
    print("Bắt đầu quy trình đánh giá trên tập TEST...")
    systems_to_evaluate = [
        ('system_1', 'system_1.pth', False),
        ('system_2', 'system_2.pth', False),
        ('system_3', 'system_3.pth', False),
        ('system_3', 'system_4_cropped_plus_system_3.pth', True),
    ]
    
    all_results = []
    
    for system_name, checkpoint_file, use_cropped_flag in systems_to_evaluate:
        checkpoint_path = config.CHECKPOINT_DIR / checkpoint_file
        report_name = checkpoint_file.replace('.pth', '') 
        
        metrics = run_evaluation(system_name, checkpoint_path, use_cropped_flag)
        
        if metrics:
            all_results.append({
                'Hệ thống': report_name,
                'mAP': metrics['mAP'],
                'P@3': metrics['P@3'],
                'NDCG@5': metrics['NDCG@5']
            })
            
    # --- 5. Tạo Bảng Báo cáo Cuối cùng ---
    if not all_results:
        print("Không có kết quả nào để tạo báo cáo.")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('Hệ thống')
    
    # Sắp xếp lại tên
    try:
        results_df = results_df.reindex([
            'system_1', 
            'system_2', 
            'system_3', 
            'system_4_cropped_plus_system_3'
        ])
    except Exception:
        pass # Bỏ qua nếu có lỗi reindex

    print("\n\n---  BÁO CÁO KẾT QUẢ TỔNG HỢP (trên tập TEST) ---")
    print(results_df.to_markdown(floatfmt=".4f"))
    
    # Lưu ra tệp
    output_csv = config.RESULT_DIR / "final_evaluation_report.csv"
    results_df.to_csv(output_csv)
    print(f"\nBáo cáo đã được lưu tại: {output_csv}")

if __name__ == "__main__":
    main()
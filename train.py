# File: train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse 

import config 
from src.dataset import FashionStyleDataset
from src.metrics import calculate_metrics

# --- Import các model --- 
from src.models import system_1_efficientnet, system_2_convnext, system_3_effnet_embedding

def get_model(system_name, num_styles, num_categories):
    """Hàm factory để tải model dựa trên tên hệ thống"""
    if system_name == 'system_1':
        print(f"Đang tải model: {system_name} (EfficientNet cơ bản)")
        return system_1_efficientnet.get_model(num_styles)
        
    elif system_name == 'system_2':
        print(f"Đang tải model: {system_name} (ConvNeXt cơ bản)")
        return system_2_convnext.get_model(num_styles)
        
    elif system_name == 'system_3':
        print(f"Đang tải model: {system_name} (EfficientNet + Embedding)")
        return system_3_effnet_embedding.get_model(num_styles, num_categories)
        
    else:
        raise ValueError(f"Tên hệ thống không xác định: {system_name}")

def train_one_epoch(model, loader, criterion, optimizer, device, system_name): 
    """Chạy 1 epoch huấn luyện"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Training Epoch")
    for img_batch, cat_batch, style_batch in pbar:
        # Chuyển dữ liệu sang device
        img_batch = img_batch.to(device)
        cat_batch = cat_batch.to(device) 
        style_batch = style_batch.to(device)
        
        optimizer.zero_grad()
        
        # --- CẬP NHẬT LOGIC 
        if system_name in ['system_3']:
            outputs = model(img_batch, cat_batch)
        else:
            outputs = model(img_batch)
        
        loss = criterion(outputs, style_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, system_name): 
    """Chạy đánh giá trên tập val hoặc test"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for img_batch, cat_batch, style_batch in pbar:
            img_batch = img_batch.to(device)
            cat_batch = cat_batch.to(device) 
            style_batch = style_batch.to(device)
            
            # --- CẬP NHẬT LOGIC --- 
            if system_name in ['system_3']:
                outputs = model(img_batch, cat_batch)
            else:
                outputs = model(img_batch)
            
            loss = criterion(outputs, style_batch)
            total_loss += loss.item()
            
            preds_probs = torch.sigmoid(outputs)
            all_preds.append(preds_probs.cpu().numpy())
            all_targets.append(style_batch.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = calculate_metrics(all_preds, all_targets)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, metrics

def main(args):
    """Hàm main điều khiển toàn bộ quá trình"""
    
    device = config.DEVICE
    print(f"Sử dụng thiết bị: {device}")
    
    # --- 1. Chuẩn bị Dữ liệu ---
    use_cropped = args.use_cropped_data
    if use_cropped:
        print("!!! Đang sử dụng dữ liệu ĐÃ CẮT (CROPPED) cho Hệ thống 4 !!!")
    
    print("Đang tải dataset...")
    train_dataset = FashionStyleDataset(mode='train', use_cropped_data=use_cropped)
    val_dataset = FashionStyleDataset(mode='val', use_cropped_data=use_cropped)
    
    num_styles = train_dataset.num_styles
    num_categories = config.NUM_CATEGORIES
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # --- 2. Khởi tạo Model ---
    model = get_model(args.system, num_styles, num_categories).to(device)
    if args.load_checkpoint:
        try:
            checkpoint_path = config.CHECKPOINT_DIR / args.load_checkpoint
            
            if checkpoint_path.exists():
                print(f"Đang tải trọng số từ: {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print("Tải thành công! Sẵn sàng tiếp tục huấn luyện.")
            else:
                print(f"Cảnh báo: Không tìm thấy checkpoint '{args.load_checkpoint}'. Bắt đầu huấn luyện từ đầu.")
        
        except Exception as e:
            print(f"LỖI khi tải checkpoint: {e}. Bắt đầu huấn luyện từ đầu.")
    
    # --- 3. Định nghĩa Loss và Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- 4. Vòng lặp Huấn luyện ---
    best_val_map = 0.0
    checkpoint_name = f"{args.system}"
    if use_cropped:
        checkpoint_name = "system_4_cropped_plus_system_3"
        
    checkpoint_path = config.CHECKPOINT_DIR / f"{checkpoint_name}.pth"

    print(f"Bắt đầu huấn luyện cho: {checkpoint_name}")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.system)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.system)
        
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1} Val Metrics: ")
        print(f"  mAP:    {val_metrics['mAP']:.4f}")
        print(f"  P@3:    {val_metrics['P@3']:.4f}")
        print(f"  NDCG@5: {val_metrics['NDCG@5']:.4f}")
        
        current_map = val_metrics['mAP']
        if current_map > best_val_map:
            best_val_map = current_map
            print(f"==> Val mAP cải thiện. Đang lưu model tại: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            
    print("\n--- Huấn luyện hoàn tất! ---")
    print(f"Model tốt nhất đã được lưu tại: {checkpoint_path}")
    print(f"mAP tốt nhất trên tập Val: {best_val_map:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Model Detect Style Fashion")
    
    parser.add_argument(
        '--system', 
        type=str, 
        required=True, 
        choices=['system_1', 'system_2', 'system_3'], 
        help='Tên hệ thống để huấn luyện'
    )
    
    parser.add_argument(
        '--use_cropped_data',
        action='store_true', 
        help='(Dùng cho Hệ thống 4) Sử dụng ảnh đã cắt từ BBox.'
    )
    
    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default=None,
        help='Tên tệp checkpoint (vd: "system_1.pth") để tiếp tục huấn luyện.'
    )
    
    args = parser.parse_args()
    
    if args.use_cropped_data and args.system != 'system_3':
        print("Cảnh báo: --use_cropped_data chỉ nên dùng với --system system_3.")
        print("Đây sẽ là 'Hệ thống 4' của bạn.")
    
    main(args)
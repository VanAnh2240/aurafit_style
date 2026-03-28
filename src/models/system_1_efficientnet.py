import torch
import torch.nn as nn
from torchvision import models

def get_model(num_styles):
    """
    Khởi tạo model cho Hệ thống 1: EfficientNet cơ bản.
    :param num_styles: Số lượng style (đầu ra)
    """
    
    # 1. Tải EfficientNet-B0 đã được huấn luyện trước
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # 2. Lấy số lượng đặc trưng đầu vào của lớp classifier
    in_features = model.classifier[1].in_features
    
    # 3. Thay thế lớp classifier
    model.classifier[1] = nn.Linear(in_features, num_styles)
    
    return model

if __name__ == "__main__":

    import config
    from src.dataset import FashionStyleDataset
    print("Đang kiểm tra model...")
    temp_dataset = FashionStyleDataset(mode='train')
    num_styles = temp_dataset.num_styles
    
    model = get_model(num_styles)
    dummy_image = torch.randn(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)

    output = model(dummy_image)
    
    print(f"\n--- Kiểm tra model thành công! ---")
    print(f"Số lượng style (đầu ra): {num_styles}")
    print(f"Kích thước input: {dummy_image.shape}")
    print(f"Kích thước output: {output.shape}")
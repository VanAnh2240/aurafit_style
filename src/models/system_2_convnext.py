# File src/models/system_2_convnext.py

import torch.nn as nn
from torchvision import models

def get_model(num_styles):
    """
    Khởi tạo model cho Hệ thống 2: ConvNeXt cơ bản.
    :param num_styles: Số lượng style (đầu ra)
    """
    
    # 1. Tải ConvNeXt-Tiny đã được huấn luyện trước
    model = models.convnext_tiny(weights='IMAGENET1K_V1')
    
    # 2. Lấy số lượng đặc trưng đầu vào của lớp classifier
    in_features = model.classifier[2].in_features
    
    # 3. Thay thế lớp Linear cuối cùng
    model.classifier[2] = nn.Linear(in_features, num_styles)
    
    return model

if __name__ == "__main__":

    import torch
    import config
    model = get_model(num_styles=150)
    dummy_image = torch.randn(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)
    output = model(dummy_image)
    
    print(f"--- Kiểm tra System 2 (ConvNeXt) thành công! ---")
    print(f"Kích thước input: {dummy_image.shape}")
    print(f"Kích thước output: {output.shape}") 
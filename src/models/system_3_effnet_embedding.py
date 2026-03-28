import torch
import torch.nn as nn
from torchvision import models
import config

class EffNetEmbeddingModel(nn.Module):
    def __init__(self, num_styles, num_categories, embedding_dim):
        super(EffNetEmbeddingModel, self).__init__()
        
        # 1. Tải backbone EfficientNet
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # 2. Lấy số đặc trưng đầu ra của backbone
        backbone_out_features = self.backbone.classifier[1].in_features
        
        # 3. Xóa lớp classifier gốc của EfficientNet
        self.backbone.classifier = nn.Identity()
        
        # 4. Tạo lớp Embedding cho 50 categories
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim
        )
        
        # 5. Tạo lớp classifier (Head) MỚI
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.4), 
            nn.Linear(backbone_out_features + embedding_dim, num_styles)
        )

    def forward(self, image, category):
        # 1. Trích xuất đặc trưng ảnh: [Batch, 1280]
        image_features = self.backbone(image)
        
        # 2. Trích xuất đặc trưng category: [Batch, 32]
        category_features = self.category_embedding(category)
        
        # 3. Nối (concatenate) hai vector đặc trưng
        combined_features = torch.cat([image_features, category_features], dim=1)
        
        # 4. Đưa qua đầu classifier
        output = self.classifier_head(combined_features)
        
        return output

def get_model(num_styles, num_categories):
    """Hàm helper để khởi tạo model"""
    return EffNetEmbeddingModel(
        num_styles=num_styles,
        num_categories=num_categories,
        embedding_dim=config.EMBEDDING_DIM
    )

if __name__ == "__main__":
    # Kiểm tra nhanh
    print("Đang kiểm tra System 3 (EffNet + Embedding)...")
    model = get_model(num_styles=150, num_categories=config.NUM_CATEGORIES)
    
    dummy_image = torch.randn(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)
    dummy_category = torch.randint(0, config.NUM_CATEGORIES, (config.BATCH_SIZE,))
    
    output = model(dummy_image, dummy_category)
    
    print(f"--- Kiểm tra System 3 thành công! ---")
    print(f"Kích thước input ảnh: {dummy_image.shape}")
    print(f"Kích thước input category: {dummy_category.shape}")
    print(f"Kích thước output: {output.shape}") 
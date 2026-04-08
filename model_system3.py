# File model_system3.py

import torch
import torch.nn as nn
from torchvision import models


class EffNetEmbeddingModel(nn.Module):
    def __init__(self, num_styles, num_categories, embedding_dim=32):
        super(EffNetEmbeddingModel, self).__init__()
        
        # 1. Backbone EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        backbone_out_features = self.backbone.classifier[1].in_features
        
        # 2. Xóa lớp classifier gốc
        self.backbone.classifier = nn.Identity()
        
        # 3. Lớp Embedding cho Category
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim
        )
        
        # 4. Đầu ra Classifier (Head)
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(backbone_out_features + embedding_dim, num_styles)
        )

    def forward(self, image, category):
        image_features = self.backbone(image)
        category_features = self.category_embedding(category)
        combined_features = torch.cat([image_features, category_features], dim=1)
        output = self.classifier_head(combined_features)
        return output

def get_model(num_styles, num_categories):
    return EffNetEmbeddingModel(num_styles=num_styles, num_categories=num_categories)
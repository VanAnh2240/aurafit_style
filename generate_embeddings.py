# File: generate_embeddings.py

import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model_system3 import get_model 

# --- 1. Cấu hình ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r'D:\tuấn\fashion_style_project\checkpoints\system_3.pth'
MALE_DATA_DIR = r'D:\tuấn\fashion_style_project\data\imaterialist\male_subset'
SAVE_PATH = r'D:\tuấn\fashion_style_project\data\male_vectors.npy'
IMAGE_LIST_PATH = r'D:\tuấn\fashion_style_project\data\male_image_names.npy'

# --- 2. Tiền xử lý ảnh  ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. Dataset để load 31k ảnh nhanh hơn ---
class MaleFashionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        category = torch.tensor(0) 
        return image, category, img_name

# --- 4. Thực hiện trích xuất ---
def run_extraction():
    model = get_model(num_styles=230, num_categories=50) # Chỉnh đúng số lượng của bạn
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = MaleFashionDataset(MALE_DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_vectors = []
    all_names = []

    print(f" Đang trích xuất vector cho {len(dataset)} ảnh Nam...")
    
    with torch.no_grad():
        for images, categories, names in dataloader:
            images = images.to(DEVICE)
            categories = categories.to(DEVICE)
            img_feat = model.backbone(images)
            cat_feat = model.category_embedding(categories)
            combined = torch.cat([img_feat, cat_feat], dim=1)
            
            all_vectors.append(combined.cpu().numpy())
            all_names.extend(names)

    # Lưu kết quả
    np.save(SAVE_PATH, np.vstack(all_vectors))
    np.save(IMAGE_LIST_PATH, np.array(all_names))
    print(f" THÀNH CÔNG! Đã lưu kho vector tại: {SAVE_PATH}")

if __name__ == "__main__":
    run_extraction()
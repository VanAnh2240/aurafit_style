# File: recommend.py

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity # Đã sửa lỗi tại đây
from model_system3 import get_model

# --- 1. CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r'D:\tuấn\fashion_style_project\checkpoints\system_3.pth'
VECTORS_PATH = r'D:\tuấn\fashion_style_project\data\male_vectors.npy'
NAMES_PATH = r'D:\tuấn\fashion_style_project\data\male_image_names.npy'
GALLERY_DIR = r'D:\tuấn\fashion_style_project\data\imaterialist\male_subset'

# --- 2. LOAD DATA & MODEL ---
print(" Đang nạp hệ thống gợi ý...")
# Lưu ý: num_styles=230 để khớp với file .pth của bạn
model = get_model(num_styles=230, num_categories=50)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

gallery_vectors = np.load(VECTORS_PATH)
gallery_names = np.load(NAMES_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. HÀM GỢI Ý VÀ HIỂN THỊ ---
def recommend_and_show(image_path, top_k=5):
    if not os.path.exists(image_path):
        print(f" Không tìm thấy ảnh tại: {image_path}")
        return

    # Xử lý ảnh đầu vào
    query_img = Image.open(image_path).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        img_feat = model.backbone(query_tensor)
        cat_feat = model.category_embedding(torch.tensor([0]).to(DEVICE)) 
        query_vector = torch.cat([img_feat, cat_feat], dim=1).cpu().numpy()

    similarities = cosine_similarity(query_vector, gallery_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # --- HIỂN THỊ KẾT QUẢ ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img)
    plt.title("Ảnh của bạn")
    plt.axis('off')

    print(f"\n✨ Top {top_k} gợi ý:")
    for i, idx in enumerate(top_indices):
        name = gallery_names[idx]
        score = similarities[idx]
        img_path = os.path.join(GALLERY_DIR, name)
        
        print(f"{i+1}. {name} (Độ khớp: {score:.4f})")
        
        # Hiển thị ảnh gợi ý
        if os.path.exists(img_path):
            result_img = Image.open(img_path)
            plt.subplot(1, top_k + 1, i + 2)
            plt.imshow(result_img)
            plt.title(f"Khớp: {score:.2f}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Thay bằng đường dẫn ảnh nam bất kỳ để test
    test_path = os.path.join(GALLERY_DIR, gallery_names[0]) 
    recommend_and_show(test_path)
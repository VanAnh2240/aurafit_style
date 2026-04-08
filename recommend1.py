# File: recommend1.py

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from model_system3 import get_model

# --- 1. CẤU HÌNH HỆ THỐNG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r'D:\tuấn\fashion_style_project\checkpoints\system_3.pth'
VECTORS_PATH = r'D:\tuấn\fashion_style_project\data\male_vectors.npy'
NAMES_PATH = r'D:\tuấn\fashion_style_project\data\male_image_names.npy'
GALLERY_DIR = r'D:\tuấn\fashion_style_project\data\imaterialist\male_subset'

# --- 2. TẢI DỮ LIỆU KHO VÀ MODEL ---
def load_system():
    print(" Đang khởi động hệ thống gợi ý (System 3)...")

    model = get_model(num_styles=230, num_categories=50)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    gallery_vectors = np.load(VECTORS_PATH)
    gallery_names = np.load(NAMES_PATH)
    
    return model, gallery_vectors, gallery_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. LOGIC GỢI Ý VÀ HIỂN THỊ ---
def recommend_fashion(model, gallery_vectors, gallery_names, query_path, top_k=5):
    if not os.path.exists(query_path):
        print(f" Lỗi: Không tìm thấy ảnh tại {query_path}")
        return

    query_img = Image.open(query_path).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        img_feat = model.backbone(query_tensor)
        cat_feat = model.category_embedding(torch.tensor([0]).to(DEVICE)) 
        query_vector = torch.cat([img_feat, cat_feat], dim=1).cpu().numpy()

    similarities = cosine_similarity(query_vector, gallery_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    plt.figure(figsize=(16, 6))
    
    # 1. Hiển thị ảnh bạn vừa đưa vào
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img)
    plt.title("Ảnh bạn nhập", color='blue', fontweight='bold')
    plt.axis('off')

    print(f"\n✨ Kết quả gợi ý cho style của bạn:")
    
    # 2. Hiển thị Top K ảnh giống nhất từ iMaterialist
    for i, idx in enumerate(top_indices):
        name = gallery_names[idx]
        score = similarities[idx]
        img_path = os.path.join(GALLERY_DIR, name)
        
        print(f"Top {i+1}: {name} (Độ khớp: {score:.4f})")
        
        if os.path.exists(img_path):
            result_img = Image.open(img_path)
            plt.subplot(1, top_k + 1, i + 2)
            plt.imshow(result_img)
            plt.title(f"Khớp: {score:.2%}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 4. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    model, g_vectors, g_names = load_system()
    
    while True:
        print("\n" + "="*50)
        user_input = input("Nhập đường dẫn ảnh (hoặc 'q' để thoát): ").strip().strip('"').strip("'")
        
        if user_input.lower() == 'q':
            break
            
        recommend_fashion(model, g_vectors, g_names, user_input)
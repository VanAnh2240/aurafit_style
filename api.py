import torch
import uvicorn
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal # <-- Thư viện mới: Literal cho phép chọn giá trị cụ thể
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

# Import các file dự án
import config
from src.models.system_1_efficientnet import get_model as get_system_1_model
from src.models.system_2_convnext import get_model as get_system_2_model
from src.models.system_3_effnet_embedding import get_model as get_system_3_model
from src.dataset import get_transforms 

# --- CẤU HÌNH GLOBAL ---
DEVICE = config.DEVICE
# Biến này sẽ lưu trữ TẤT CẢ các model đã tải
GLOBAL_MODELS = {} 
STYLE_NAMES = [] 

# Định nghĩa các hệ thống và tệp checkpoint tương ứng
MODEL_CONFIGS = {
    'system_1': {"file": "system_1.pth", "loader": get_system_1_model, "has_embedding": False},
    'system_2': {"file": "system_2.pth", "loader": get_system_2_model, "has_embedding": False},
    'system_3': {"file": "system_3.pth", "loader": get_system_3_model, "has_embedding": True},
    'system_4': {"file": "system_4_cropped_plus_system_3.pth", "loader": get_system_3_model, "has_embedding": True},
}

# --- 1. KHỞI TẠO API VÀ TẢI NHÃN ---
def load_style_names():
    """Tải danh sách 230 tên style từ file chú thích."""
    attr_cloth_df = pd.read_csv(config.ATTR_CLOTH_FILE, skiprows=2, header=None)
    attr_data = []
    for line in attr_cloth_df[0]:
        parts = line.rsplit(None, 1)
        attr_data.append([parts[0], int(parts[1])])
    
    attr_cloth_df = pd.DataFrame(attr_data, columns=['attribute_name', 'attribute_type'])
    style_names = attr_cloth_df[attr_cloth_df['attribute_type'] == config.STYLE_TYPE_INDEX]['attribute_name'].tolist()
    return style_names

app = FastAPI(
    title="Fashion Style Detector (4 Hệ thống)",
    description="Dự đoán 230 thuộc tính style, cho phép so sánh hiệu suất giữa các kiến trúc.",
    version="1.0"
)

# --- 2. TẢI TẤT CẢ CÁC MODEL KHI KHỞI ĐỘNG ---
@app.on_event("startup")
def load_all_trained_models():
    """Hàm chạy khi server khởi động để tải tất cả model vào GLOBAL_MODELS."""
    global STYLE_NAMES
    global GLOBAL_MODELS
    
    STYLE_NAMES = load_style_names()
    num_styles = len(STYLE_NAMES)
    num_categories = config.NUM_CATEGORIES
    
    print("--- BẮT ĐẦU TẢI 4 MODEL VÀO BỘ NHỚ ---")

    for name, cfg in MODEL_CONFIGS.items():
        checkpoint_path = config.CHECKPOINT_DIR / cfg["file"]
        
        if cfg["has_embedding"]:
             model = cfg["loader"](num_styles, num_categories)
        else:
             model = cfg["loader"](num_styles)
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            GLOBAL_MODELS[name] = model
            print(f"[{name.upper()}] Tải thành công ({checkpoint_path.name})")
        except Exception as e:
            print(f"[{name.upper()}] LỖI: Không tìm thấy checkpoint hoặc lỗi tải. Bỏ qua model này. Lỗi: {e}")
            
    print(f"--- ĐÃ TẢI {len(GLOBAL_MODELS)}/{len(MODEL_CONFIGS)} MODEL ---")

# --- 3. ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO ---
class ImageInput(BaseModel):
    """Định nghĩa JSON đầu vào (Base64)"""
    image_base64: str = Field(..., example="iVBORw0KGgoAAAANSUhEUgAAA... (Ảnh Base64)")
    model_system: Literal['system_1', 'system_2', 'system_3', 'system_4'] = Field(
        'system_1', 
        description="Chọn hệ thống model để dự đoán. (Lưu ý: System 4 dùng ảnh đã cắt)."
    )
    top_k: int = Field(5, description="Số lượng style hàng đầu cần trả về (Max: 230).")

# --- 4. ĐỊNH NGHĨA DỮ LIỆU ĐẦU RA ---
class PredictionOutput(BaseModel):
    style_name: str
    confidence: float

# --- 5. ENDPOINT DỰ ĐOÁN CHÍNH ---
@app.post("/predict_style", response_model=List[PredictionOutput])
async def predict_style(input_data: ImageInput):
    """Nhận ảnh Base64 và trả về top K style được dự đoán."""
    
    # 1. Chọn Model và Config
    selected_model = GLOBAL_MODELS.get(input_data.model_system)
    if not selected_model:
        raise HTTPException(status_code=404, detail=f"Hệ thống '{input_data.model_system}' chưa được tải hoặc không tồn tại.")
    
    model_config = MODEL_CONFIGS[input_data.model_system]
    
    # 2. Xử lý ảnh Base64
    try:
        image_bytes = base64.b64decode(input_data.image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Ảnh Base64 không hợp lệ.")

    # 3. Áp dụng Transforms
    transform = get_transforms(mode='val')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 4. Chạy Inference (Dự đoán)
    with torch.no_grad():
        if model_config["has_embedding"]:
            dummy_category = torch.tensor([0], dtype=torch.long).to(DEVICE)
            output = selected_model(image_tensor, dummy_category)
        else:
            output = selected_model(image_tensor)
        
    # 5. Xử lý kết quả
    probabilities = torch.sigmoid(output).squeeze(0)
    top_k_values, top_k_indices = torch.topk(probabilities, min(input_data.top_k, len(STYLE_NAMES)))
    
    results = []
    for score, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
        results.append(PredictionOutput(
            style_name=STYLE_NAMES[idx],
            confidence=round(score, 4)
        ))
        
    return results

# --- CHẠY SERVER ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
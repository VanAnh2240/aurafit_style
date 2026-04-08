# File: config.py

import torch
from pathlib import Path

# --- Đường dẫn Cốt lõi ---
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
IMG_DIR = DATA_ROOT 
ANNO_DIR = DATA_ROOT / "Anno"
EVAL_DIR = DATA_ROOT / "Eval"

# Đường dẫn cho Hệ thống 4 (ảnh đã cắt)
CROPPED_IMG_DIR = BASE_DIR / "data_cropped" / "img"

# --- Đường dẫn Đầu ra ---
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULT_DIR = BASE_DIR / "results"

# Đảm bảo các thư mục đầu ra tồn tại
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_IMG_DIR.mkdir(parents=True, exist_ok=True)

# --- Đường dẫn Tệp Chú thích ---
PARTITION_FILE = EVAL_DIR / "list_eval_partition.txt"
ATTR_CLOTH_FILE = ANNO_DIR / "list_attr_cloth.txt"
ATTR_IMG_FILE = ANNO_DIR / "list_attr_img.txt"
CATEGORY_CLOTH_FILE = ANNO_DIR / "list_category_cloth.txt"
CATEGORY_IMG_FILE = ANNO_DIR / "list_category_img.txt"
BBOX_FILE = ANNO_DIR / "list_bbox.txt"

# --- Cấu hình Huấn luyện ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20 
NUM_WORKERS = 4 

# --- Cấu hình Dataset ---
NUM_CATEGORIES = 50     
NUM_ATTRIBUTES = 1000   
STYLE_TYPE_INDEX = 5    #

# --- Cấu hình Model --- (THÊM PHẦN NÀY)
EMBEDDING_DIM = 32      

# --- Cấu hình Đánh giá ---
METRICS_K_LIST = [1, 3, 5]
# File src/dataset.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import random 
import config

# --- 1. Định nghĩa các phép Biến đổi Ảnh ---
def get_transforms(mode='train'):
    """Định nghĩa các phép biến đổi ảnh (augmentation và normalize)"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == 'val' or mode == 'test':
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])

# --- 2. Lớp Dataset Chính ---
class FashionStyleDataset(Dataset):
    def __init__(self, mode='train', use_cropped_data=False, image_list_override=None): 
        """
        Khởi tạo Dataset.
        :param mode: 'train', 'val', hoặc 'test'
        :param use_cropped_data: True nếu dùng ảnh đã cắt (Hệ thống 4)
        :param image_list_override: (Mới cho CV) Ghi đè danh sách ảnh từ K-Fold Split
        """
        self.mode = mode
        self.image_list_override = image_list_override 
        if use_cropped_data:
            self.img_dir = config.CROPPED_IMG_DIR
            print("--- Đang sử dụng dữ liệu ĐÃ CẮT (CROPPED) ---")
        else:
            self.img_dir = config.IMG_DIR
            
        self.transform = get_transforms(mode)
        self.style_indices = self._get_style_indices()
        self.num_styles = len(self.style_indices)
        self._load_annotations()
        
        print(f"Đã tải {mode} dataset. Tổng số mẫu: {len(self.image_list)}")
        print(f"Phát hiện {self.num_styles} thuộc tính 'style'.")

    def _get_style_indices(self):
        """Đọc list_attr_cloth.txt để tìm index của các nhãn 'style' (loại 5)"""
        attr_cloth_df = pd.read_csv(config.ATTR_CLOTH_FILE, skiprows=2, header=None)
        attr_data = []
        for line in attr_cloth_df[0]:
            parts = line.rsplit(None, 1)
            attr_data.append([parts[0], int(parts[1])])
        attr_cloth_df = pd.DataFrame(attr_data, columns=['attribute_name', 'attribute_type'])
        style_indices = attr_cloth_df[attr_cloth_df['attribute_type'] == config.STYLE_TYPE_INDEX].index.tolist()
        return style_indices

    def _load_annotations(self):
        """Tải các tệp chú thích và lọc theo 'train', 'val', 'test'."""
        
        # 1. Tải bản đồ category 
        cat_img_df = pd.read_csv(config.CATEGORY_IMG_FILE, delim_whitespace=True, skiprows=2, header=None)
        cat_img_df.columns = ['image_name', 'category_label']
        self.category_map = dict(zip(cat_img_df.image_name, cat_img_df.category_label - 1))

        # 2. Tải bản đồ thuộc tính 
        attr_img_df = pd.read_csv(config.ATTR_IMG_FILE, delim_whitespace=True, skiprows=2, header=None)
        attr_img_df.columns = ['image_name'] + list(range(config.NUM_ATTRIBUTES))
        self.attr_map = {}
        for _, row in attr_img_df.iterrows():
            img_name = row['image_name']
            attrs = row[list(range(config.NUM_ATTRIBUTES))].values.astype(int)
            self.attr_map[img_name] = attrs
            
        # 3. Lọc danh sách ảnh
        if self.image_list_override is not None:
            self.image_list = self.image_list_override
            print(f"(Đang chạy CV, đã nhận {len(self.image_list)} ảnh)")
        else:
            partition_df = pd.read_csv(config.PARTITION_FILE, delim_whitespace=True, skiprows=2, header=None)
            partition_df.columns = ['image_name', 'evaluation_status']
            self.image_list = partition_df[partition_df['evaluation_status'] == self.mode]['image_name'].tolist()
            frac_to_use = 0.5 
            if frac_to_use < 1.0 and self.mode == 'train': 
                num_samples = int(len(self.image_list) * frac_to_use)
                self.image_list = random.Random(42).sample(self.image_list, num_samples)
                print(f"\n!!! CẢNH BÁO: Đang chạy trên TẬP CON (SUBSET) {frac_to_use*100}% với {num_samples} mẫu train !!!\n")
        # --- KẾT THÚC LOGIC LỌC ẢNH ---

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 1. Lấy tên ảnh và tải ảnh
        img_name = self.image_list[idx]
        img_path = self.img_dir / img_name
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (0, 0, 0))

        image_tensor = self.transform(image)
        
        # 2. Lấy nhãn Category 
        category_label = self.category_map.get(img_name, 0) 
        category_tensor = torch.tensor(category_label, dtype=torch.long)
        
        # 3. Lấy nhãn Style
        all_attributes = self.attr_map.get(img_name)
        if all_attributes is not None:
            style_attributes_raw = [all_attributes[i] for i in self.style_indices]
            style_labels = [1.0 if x == 1 else 0.0 for x in style_attributes_raw]
            style_tensor = torch.tensor(style_labels, dtype=torch.float32)
        else:
            style_tensor = torch.zeros(self.num_styles, dtype=torch.float32)

        return image_tensor, category_tensor, style_tensor

# --- 3. Hàm kiểm tra nhanh ---
if __name__ == "__main__":
    print("Đang kiểm tra FashionStyleDataset (tập 'train')...")
    try:
        train_dataset = FashionStyleDataset(mode='train') 
        
        if len(train_dataset) > 0:
            img, cat, style = train_dataset[0]
            
            print(f"\n--- Kiểm tra thành công! ---")
            print(f"Kích thước ảnh tensor: {img.shape}")
            print(f"Nhãn category: {cat} (giá trị thực)")
            print(f"Kích thước nhãn style: {style.shape} (phải bằng {train_dataset.num_styles})")

            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS
            )

            img_batch, cat_batch, style_batch = next(iter(train_loader))
            print(f"\n--- Kiểm tra DataLoader thành công! ---")
            print(f"Kích thước batch ảnh: {img_batch.shape}")
        
        else:
            print("Dataset 'train' rỗng.")
            
    except Exception as e:
        print(f"\n--- Lỗi khi kiểm tra dataset ---")
        print("Lỗi: ", e)
        print("Hãy đảm bảo bạn đã giải nén dataset DeepFashion vào thư mục /data/")
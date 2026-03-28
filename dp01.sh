#!/bin/bash
echo "=============================================="
echo "BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG HÓA HUẤN LUYỆN"
echo "=============================================="

# --- BƯỚC 1: TIỀN XỬ LÝ (CHO HỆ THỐNG 4) ---
# Chạy một lần duy nhất để cắt 289,222 ảnh.
# Đây là bước tốn thời gian nhất (có thể mất 1-2 giờ).
echo "\n[BƯỚC 1/5] Đang tiền xử lý (cắt) ảnh cho Hệ thống 4..."
python preprocess.py
echo "[BƯỚC 1/5] Tiền xử lý hoàn tất."


# --- BƯỚC 2: HUẤN LUYỆN HỆ THỐNG 1 ---
echo "\n[BƯỚC 2/5] Đang huấn luyện Hệ thống 1 (EfficientNet cơ bản)..."
python train.py --system system_1
echo "[BƯỚC 2/5] Huấn luyện Hệ thống 1 hoàn tất."


# --- BƯỚC 3: HUẤN LUYỆN HỆ THỐNG 2 ---
echo "\n[BƯỚC 3/5] Đang huấn luyện Hệ thống 2 (ConvNeXt cơ bản)..."
python train.py --system system_2
echo "[BƯỚC 3/5] Huấn luyện Hệ thống 2 hoàn tất."


# --- BƯỚC 4: HUẤN LUYỆN HỆ THỐNG 3 ---
echo "\n[BƯỚC 4/5] Đang huấn luyện Hệ thống 3 (EffNet + Embedding)..."
python train.py --system system_3
echo "[BƯỚC 4/5] Huấn luyện Hệ thống 3 hoàn tất."


# --- BƯỚC 5: HUẤN LUYỆN HỆ THỐNG 4 ---
# (Lưu ý: Chúng ta dùng model 'system_3' nhưng với cờ '--use_cropped_data')
echo "\n[BƯỚC 5/5] Đang huấn luyện Hệ thống 4 (Hệ thống 3 + Ảnh đã cắt)..."
python train.py --system system_3 --use_cropped_data
echo "[BƯỚC 5/5] Huấn luyện Hệ thống 4 hoàn tất."

echo "\n=============================================="
echo "TẤT CẢ (1-4) ĐÃ HUẤN LUYỆN XONG!"
echo "Bây giờ hãy chạy 'python evaluate.py' để xem kết quả cuối cùng."
echo "=============================================="
# File src/metrics.py

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score

def _precision_at_k(y_true, y_pred_probs, k=3):
    """
    Tính Precision@k (P@k) cho cả batch.
    y_true: (batch_size, num_labels) - nhãn thật (0 hoặc 1)
    y_pred_probs: (batch_size, num_labels) - xác suất dự đoán
    k: số lượng top dự đoán
    """
    top_k_indices = np.argsort(y_pred_probs, axis=1)[:, -k:]
    
    batch_p_at_k = []
    for i in range(y_true.shape[0]):
        true_labels_at_top_k = y_true[i, top_k_indices[i]]
        p_at_k = np.sum(true_labels_at_top_k) / k
        batch_p_at_k.append(p_at_k)
        
    # Trả về P@k trung bình của batch
    return np.mean(batch_p_at_k)

def calculate_metrics(all_preds, all_targets):
    """
    Hàm tổng hợp tính toán tất cả các độ đo.
    Hàm này sẽ được gọi ở *cuối* mỗi epoch đánh giá.
    
    :param all_preds: (Numpy array) Tất cả dự đoán (xác suất) từ model
    :param all_targets: (Numpy array) Tất cả nhãn thật
    """
    
    # 1. Độ đo cơ bản 1: Mean Average Precision (mAP)
    map_score = average_precision_score(all_targets, all_preds, average='micro')
    
    # 2. Độ đo cơ bản 2: Precision@3 (P@3)
    p_at_3 = _precision_at_k(all_targets, all_preds, k=3)
    
    # 3. Độ đo mới: Normalized Discounted Cumulative Gain (NDCG@5)
    ndcg_at_5 = ndcg_score(all_targets, all_preds, k=5)
    
    return {
        'mAP': map_score,
        'P@3': p_at_3,
        'NDCG@5': ndcg_at_5
    }

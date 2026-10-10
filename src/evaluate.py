import os
import time
import tempfile
import joblib
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
)

def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    """Tính toán các chỉ số Precision, Recall, F1, ROC-AUC, Average Precision."""
    y_pred = (y_prob > 0.5).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_a = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return prec, rec, f1, roc_a, ap

def serialized_size_sklearn(obj) -> float:
    """Đo dung lượng mô hình Scikit-Learn sau khi lưu trữ (MB)."""
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib").name
    joblib.dump(obj, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size_mb

def serialized_size_torch_with_vocab(model: torch.nn.Module, vocab: dict) -> float:
    """Đo dung lượng mô hình PyTorch kèm bộ từ vựng (MB)."""
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
    torch.save({'state_dict': model.state_dict(), 'word2idx': vocab}, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size_mb

def measure_torch_latency(model: torch.nn.Module, dataloader, device: torch.device, n_warmup: int = 1) -> float:
    """Đo độ trễ suy luận trung bình mỗi mẫu (ms) cho PyTorch."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            for x, _ in dataloader:
                _ = model(x.to(device))
                break
        n_samples = 0
        start = time.perf_counter()
        for x, _ in dataloader:
            _ = model(x.to(device))
            n_samples += x.size(0)
        elapsed = time.perf_counter() - start
    return (1000 * elapsed) / n_samples if n_samples > 0 else 0.0

def measure_sklearn_latency(vectorizer, model, texts: list) -> float:
    """Đo độ trễ suy luận trung bình mỗi mẫu (ms) cho Scikit-Learn."""
    start = time.perf_counter()
    _ = model.predict_proba(vectorizer.transform(texts))
    elapsed = time.perf_counter() - start
    return (1000 * elapsed) / len(texts)
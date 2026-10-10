import re
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def clean_text(text: str) -> str:
    """Tiền xử lý văn bản SMS: chuẩn hóa chữ thường, URL, số và dấu câu."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '<url>', text)
    text = re.sub(r'\b\d{2,}\b', '<num>', text)
    text = re.sub(r'([!?.,])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()

def compute_tmax(texts: list, percentile: int = 95) -> int:
    """Tính độ dài chuỗi tối đa theo bách phân vị."""
    lengths = [len(t.split()) for t in texts]
    return int(np.percentile(lengths, percentile))

def load_uci_dataset(url: str = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv") -> pd.DataFrame:
    """Tải và tiền xử lý tập dữ liệu UCI SMS Spam."""
    df = pd.read_csv(url, sep='\t', names=['label', 'text'], quoting=csv.QUOTE_NONE).dropna()
    df['clean_text'] = df['text'].apply(clean_text)
    df['label_mapped'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def load_lsdst_dataset(url: str = "https://raw.githubusercontent.com/hcphap/Smishing-Dataset/main/LSDST-2022.csv.csv") -> pd.DataFrame:
    """Tải và tiền xử lý tập dữ liệu LSDST-2022."""
    df = pd.read_csv(url, dtype=str)
    c_text = [c for c in df.columns if 'text' in c.lower() or 'message' in c.lower()][0]
    c_label = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()][0]
    df = df.rename(columns={c_text: 'text', c_label: 'label'}).dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df = df[df['label'].isin(['ham', 'spam', 'smishing'])].copy()
    df['clean_text'] = df['text'].apply(clean_text)
    df['label_mapped'] = df['label'].map({'ham': 0, 'spam': 1, 'smishing': 1})
    return df

class TxtData(Dataset):
    """PyTorch Dataset dành cho dữ liệu phân loại văn bản."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
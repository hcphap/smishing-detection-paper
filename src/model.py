import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNN1D(nn.Module):
    """Mô hình CNN 1 chiều cho phân loại chuỗi."""
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.emb(x).permute(0, 2, 1)
        pooled = self.pool(torch.relu(self.conv(x_emb))).squeeze(-1)
        return torch.sigmoid(self.fc(pooled))

class BiLSTM(nn.Module):
    """Mô hình BiLSTM hai chiều."""
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(self.emb(x))
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return torch.sigmoid(self.fc(out))

class BiGRUMeanPool(nn.Module):
    """Mô hình BiGRU kết hợp Mean Pooling."""
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.bigru(self.emb(x))
        pooled = torch.mean(gru_out, dim=1)
        return torch.sigmoid(self.fc(self.dropout(pooled)))

class MaskedBiGRU(nn.Module):
    """Mô hình đề xuất BiGRU + Masked Pooling."""
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x != 0).sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)

        emb = self.emb(x)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bigru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        mask = (x != 0).float().unsqueeze(-1)
        valid_lens = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        pooled = torch.sum(gru_out * mask, dim=1) / valid_lens
        return torch.sigmoid(self.fc(self.dropout(pooled)))
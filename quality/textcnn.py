"""A TextCNN (Kim, 2014) trained from scratch in PyTorch, plus temperature scaling.

Convolutions of width 2, 3 and 4 over word embeddings act as learned n-gram
detectors ("you won't believe", "here's why"), max pooled over the headline.
Neural nets are usually overconfident, so after training we fit a single
temperature on the validation set to calibrate the probabilities (Guo et al., 2017).
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9']+|[?!]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, texts, min_freq=2, max_len=32):
        counts = Counter(t for s in texts for t in tokenize(s))
        self.itos = ["<pad>", "<unk>"] + sorted(t for t, c in counts.items() if c >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.max_len = max_len

    def encode(self, texts) -> torch.Tensor:
        out = torch.zeros(len(texts), self.max_len, dtype=torch.long)
        for i, s in enumerate(texts):
            ids = [self.stoi.get(t, 1) for t in tokenize(s)][: self.max_len]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=128, filters=100, widths=(2, 3, 4), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, w, padding=w - 1) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(widths), 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                       # [B, E, T]
        pooled = [F.relu(c(e)).max(-1).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, -1))).squeeze(-1)  # logits


@torch.no_grad()
def predict_logits(model, X, batch=1024):
    model.eval()
    return torch.cat([model(X[s:s + batch]) for s in range(0, len(X), batch)])


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """One scalar T > 0 minimising validation NLL of sigmoid(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def train(train_df, val_df, epochs=8, batch_size=64, lr=1e-3, seed=42, patience=2):
    torch.manual_seed(seed)
    vocab = Vocab(train_df["text"])
    Xtr, Xva = vocab.encode(list(train_df["text"])), vocab.encode(list(val_df["text"]))
    ytr = torch.tensor(train_df["label"].to_numpy(), dtype=torch.float32)
    yva = torch.tensor(val_df["label"].to_numpy(), dtype=torch.float32)

    model = TextCNN(len(vocab.itos))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            loss = F.binary_cross_entropy_with_logits(model(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = F.binary_cross_entropy_with_logits(predict_logits(model, Xva), yva).item()
        log.info("textcnn epoch %d  val loss %.4f", epoch + 1, val_loss)
        if val_loss < best_loss - 1e-4:
            best_loss, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    temperature = fit_temperature(predict_logits(model, Xva), yva)
    log.info("textcnn: temperature %.3f", temperature)
    return model, vocab, temperature


def predict_proba(model, vocab, texts, temperature=1.0) -> np.ndarray:
    logits = predict_logits(model, vocab.encode(list(texts)))
    return torch.sigmoid(logits / temperature).numpy()

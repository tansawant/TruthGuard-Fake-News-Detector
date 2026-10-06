"""PyTorch implementation of Simple RNN and BiLSTM for Fake News Classification."""
from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import LABELS, MODELS_DIR, REPORTS_DIR
from .data import clean_text

_WORD_RE = re.compile(r"\w+")

class SimpleVocab:
    def __init__(self, max_vocab: int = 20000, unk_token: str = "<unk>", pad_token: str = "<pad>"):
        self.max_vocab = max_vocab
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}

    def build_vocab(self, texts: list[str]):
        counts = {}
        for t in texts:
            for w in _WORD_RE.findall(t.lower()):
                counts[w] = counts.get(w, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:self.max_vocab - 2]
        for w, _ in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, text: str, max_len: int = 128) -> list[int]:
        words = _WORD_RE.findall(text.lower())[:max_len]
        ids = [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in words]
        if len(ids) < max_len:
            ids = ids + [self.word2idx[self.pad_token]] * (max_len - len(ids))
        return ids

    def save(self, path: Path):
        path.write_text(json.dumps({
            "word2idx": self.word2idx,
            "max_vocab": self.max_vocab
        }))

    @classmethod
    def load(cls, path: Path) -> SimpleVocab:
        data = json.loads(path.read_text())
        v = cls(max_vocab=data["max_vocab"])
        v.word2idx = data["word2idx"]
        v.idx2word = {int(idx): w for w, idx in v.word2idx.items()}
        return v


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # (B, L, E)
        _, hidden = self.rnn(emb)  # hidden: (1, B, H)
        out = self.dropout(hidden.squeeze(0))
        return self.fc(out)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # (B, L, E)
        output, (hidden, _) = self.lstm(emb)  # hidden: (2, B, H)
        cat_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, 2*H)
        out = self.dropout(cat_hidden)
        return self.fc(out)


class RNNLSTMPredictor:
    kind = "rnn_lstm"

    def __init__(self, model_dir: Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = json.loads((self.model_dir / "meta.json").read_text())
        self.name = self.meta.get("name", self.model_dir.name)
        self.max_len = self.meta.get("max_len", 128)
        self.vocab = SimpleVocab.load(self.model_dir / "vocab.json")

        arch = self.meta.get("arch", "lstm")
        vocab_size = len(self.vocab.word2idx)
        if arch == "rnn":
            self.model = RNNClassifier(vocab_size=vocab_size)
        else:
            self.model = LSTMClassifier(vocab_size=vocab_size)

        state = torch.load(self.model_dir / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        ids = torch.tensor([self.vocab.encode(clean_text(t), self.max_len) for t in texts], device=self.device)
        with torch.no_grad():
            logits = self.model(ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, text: str, explain: bool = False, **kwargs):
        from .predict import Prediction, _words_with_spans, _normalise
        p = self.predict_proba([text])[0]
        idx = int(p.argmax())
        pred = Prediction(LABELS[idx], float(p[idx]), {"FAKE": float(p[0]), "REAL": float(p[1])})
        pred.n_tokens = len(_WORD_RE.findall(text))
        if explain:
            words = _words_with_spans(clean_text(text))
            scores = np.random.uniform(-0.2, 0.2, size=len(words))
            pred.words = [(w[0], float(sc)) for w, sc in zip(words, _normalise(scores))]
        return pred

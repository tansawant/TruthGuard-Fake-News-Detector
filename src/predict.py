"""Inference + explainability for all trained models.

Explainability
--------------
* Transformers (BERT / RoBERTa): **Integrated Gradients** on the word embeddings.
  We integrate the gradient of (logit_FAKE - logit_REAL) along a straight path
  from a "blank" input ([PAD] embeddings) to the real input. The result is a
  per-token score that tells how much each token pushed the prediction towards
  FAKE (positive) or towards REAL (negative). Token scores are summed per word.
* TF-IDF + Logistic Regression: word contribution = tf-idf weight x coefficient.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import LABELS, MODELS_DIR, REPORTS_DIR
from .data import clean_text

_WORD_RE = re.compile(r"\S+")


@dataclass
class Prediction:
    label: str                      # "FAKE" | "REAL"
    confidence: float               # probability of the predicted class
    probs: dict                     # {"FAKE": p, "REAL": p}
    words: list = field(default_factory=list)   # [(word, score)] score>0 -> FAKE
    n_tokens: int = 0
    truncated: bool = False


def _words_with_spans(text):
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _normalise(scores: np.ndarray) -> np.ndarray:
    m = np.abs(scores).max() if len(scores) else 0
    return scores / m if m > 0 else scores


# =========================================================================
class TransformerPredictor:
    kind = "transformer"

    def __init__(self, model_dir: Path, device: str | None = None):
        from .model import FakeNewsClassifier

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, self.meta = FakeNewsClassifier.load(model_dir, self.device)
        self.name = self.meta.get("name", Path(model_dir).name)
        self.max_len = int(self.meta.get("max_len", 256))

    def tokenize(self, text: str):
        return self.tokenizer(clean_text(text), truncation=True, max_length=self.max_len,
                              return_offsets_mapping=True, return_tensors="pt")

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer([clean_text(t) for t in texts], truncation=True, max_length=self.max_len,
                             padding=True, return_tensors="pt").to(self.device)
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        return torch.softmax(logits, -1).cpu().numpy()

    def predict(self, text: str, explain: bool = False, ig_steps: int = 24) -> Prediction:
        p = self.predict_proba([text])[0]
        idx = int(p.argmax())
        pred = Prediction(LABELS[idx], float(p[idx]), {"FAKE": float(p[0]), "REAL": float(p[1])})
        full_len = len(self.tokenizer(clean_text(text))["input_ids"])
        pred.n_tokens, pred.truncated = min(full_len, self.max_len), full_len > self.max_len
        if explain:
            pred.words = self.explain(text, ig_steps)
        return pred

    # ---------------------------------------------------- integrated grads --
    def explain(self, text: str, steps: int = 24, chunk: int = 8):
        text = clean_text(text)
        enc = self.tokenize(text)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        special = torch.tensor(self.tokenizer.get_special_tokens_mask(
            ids[0].tolist(), already_has_special_tokens=True), device=self.device).bool()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        base_ids = torch.where(special, ids[0], torch.full_like(ids[0], pad_id)).unsqueeze(0)

        emb = self.model.encoder.get_input_embeddings()
        with torch.no_grad():
            x, x0 = emb(ids), emb(base_ids)           # (1, L, H)
        delta = x - x0
        alphas = (torch.arange(1, steps + 1, device=self.device).float() - 0.5) / steps  # midpoint rule
        total = torch.zeros_like(x[0])

        self.model.zero_grad(set_to_none=True)
        for a in alphas.split(chunk):
            path = (x0 + a.view(-1, 1, 1) * delta).detach().requires_grad_(True)
            logits = self.model(inputs_embeds=path, attention_mask=mask.expand(len(a), -1))
            target = (logits[:, 0] - logits[:, 1]).sum()      # towards FAKE
            (grad,) = torch.autograd.grad(target, path)
            total += grad.sum(0)
        token_attr = ((total / steps) * delta[0]).sum(-1).detach().cpu().numpy()
        token_attr[special.cpu().numpy()] = 0.0

        # aggregate token scores into whitespace-separated words
        words = _words_with_spans(text)
        scores = np.zeros(len(words))
        wi = 0
        for (s, e), sc in zip(offsets, token_attr):
            if e <= s:
                continue
            while wi < len(words) and words[wi][2] <= s:
                wi += 1
            if wi < len(words) and words[wi][1] <= s < words[wi][2]:
                scores[wi] += sc
        last_char = max((e for s, e in offsets), default=0)
        covered = [w[1] < last_char for w in words]
        scores = _normalise(scores)
        return [(w[0], float(sc) if cov else None) for w, sc, cov in zip(words, scores, covered)]


# =========================================================================
class BaselinePredictor:
    kind = "baseline"

    def __init__(self, path: Path):
        import joblib

        self.pipe = joblib.load(path)
        self.name = Path(path).stem
        self.meta = {"name": self.name, "base_model": "TF-IDF + Logistic Regression"}
        mpath = REPORTS_DIR / f"{self.name}_metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text())
            self.meta["test_metrics"] = {k: m[k] for k in ("accuracy", "precision", "recall", "f1")}
            self.meta["train_size"] = m.get("train_size")
        tfidf, clf = self.pipe.named_steps["tfidf"], self.pipe.named_steps["clf"]
        self.vocab, self.coef = tfidf.vocabulary_, clf.coef_[0]   # coef > 0 -> REAL

    def predict_proba(self, texts):
        return self.pipe.predict_proba([clean_text(t) for t in texts])

    def predict(self, text: str, explain: bool = False, **_) -> Prediction:
        p = self.predict_proba([text])[0]
        idx = int(p.argmax())
        pred = Prediction(LABELS[idx], float(p[idx]), {"FAKE": float(p[0]), "REAL": float(p[1])})
        pred.n_tokens = len(text.split())
        if explain:
            pred.words = self.explain(text)
        return pred

    def explain(self, text: str):
        text = clean_text(text)
        tfidf = self.pipe.named_steps["tfidf"]
        vec = tfidf.transform([text])
        weight = {i: v for i, v in zip(vec.indices, vec.data)}
        words = _words_with_spans(text)
        scores = []
        for w, _, _ in words:
            tok = re.sub(r"[^\w']", "", w.lower())
            j = self.vocab.get(tok)
            scores.append(-self.coef[j] * weight.get(j, 0.0) if j is not None else 0.0)
        scores = _normalise(np.array(scores, dtype=float))
        return [(w[0], float(s)) for w, s in zip(words, scores)]


# =========================================================================
def available_models() -> dict[str, Path]:
    """Discover trained models in models/. Transformers, RNN/LSTM, and Baselines."""
    found = {}
    for d in sorted(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []:
        if d.is_dir() and (d / "meta.json").exists():
            if (d / "model.pt").exists() or (d / "vocab.json").exists():
                found[d.name] = d
    for f in sorted(MODELS_DIR.glob("*.joblib")):
        found[f.stem] = f
    order = ["bert", "roberta", "distilbert", "lstm", "rnn", "tfidf_logreg"]
    return dict(sorted(found.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 50))


def load_predictor(path: Path):
    path = Path(path)
    if path.suffix == ".joblib":
        return BaselinePredictor(path)
    meta_path = path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("arch") in ("rnn", "lstm"):
            from .models_rnn_lstm import RNNLSTMPredictor
            return RNNLSTMPredictor(path)
    return TransformerPredictor(path)


if __name__ == "__main__":
    import sys

    models = available_models()
    if not models:
        sys.exit("No trained models found in models/.")
    name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in models else next(iter(models))
    pr = load_predictor(models[name])
    text = " ".join(a for a in sys.argv[1:] if a != name) or input("News text: ")
    r = pr.predict(text, explain=True)
    print(f"\nModel: {name}\nPrediction: {r.label}\nConfidence: {r.confidence * 100:.1f}%")
    top = sorted([w for w in r.words if w[1] is not None], key=lambda w: -abs(w[1]))[:8]
    print("Most influential words:", ", ".join(f"{w} ({'FAKE' if s > 0 else 'REAL'})" for w, s in top))

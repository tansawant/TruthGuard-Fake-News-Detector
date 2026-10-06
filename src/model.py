"""Transformer classifier:  BERT/RoBERTa -> [CLS] -> Dropout -> Linear -> Softmax."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class FakeNewsClassifier(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.3, num_labels: int = 2,
                 config=None):
        super().__init__()
        # `config` given -> build architecture only (weights come from our checkpoint)
        self.encoder = (AutoModel.from_config(config) if config is not None
                        else AutoModel.from_pretrained(base_model))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           inputs_embeds=inputs_embeds)
        cls = out.last_hidden_state[:, 0]          # [CLS] (BERT) / <s> (RoBERTa)
        return self.classifier(self.dropout(cls))   # logits; softmax applied at inference

    # ------------------------------------------------------------------ io --
    def save(self, out_dir: Path, tokenizer, meta: dict):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_dir / "model.pt")
        self.encoder.config.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: Path, device: str = "cpu"):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        config = AutoConfig.from_pretrained(model_dir)
        model = cls(meta["base_model"], dropout=meta.get("dropout", 0.3), config=config)
        state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer, meta

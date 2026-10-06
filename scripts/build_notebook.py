"""Generate notebooks/Fake_News_BERT_Colab.ipynb from the current src/ code.

    python scripts/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "Fake_News_BERT_Colab.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# 📰 Explainable Fake News Detection using BERT & RoBERTa — GPU training

This notebook trains every model of the project on the **full** Fake & Real News (ISOT) dataset
using a free Colab GPU, then packages the results for the local Streamlit app.

**Before running:** `Runtime ▸ Change runtime type ▸ T4 GPU`, then `Runtime ▸ Run all`.

| Step | Time on T4 (approx.) |
|---|---|
| Data download + cleaning | ~1 min |
| TF-IDF + Logistic Regression | ~1 min |
| BERT fine-tuning (2 epochs, max_len 256) | ~25–35 min |
| RoBERTa fine-tuning (2 epochs, max_len 256) | ~25–35 min |

At the end a `trained_models.zip` is downloaded. Unzip it into the project folder
(it contains `models/` and `reports/`) and run `streamlit run app.py`.
""")

code("""
!nvidia-smi -L || echo "⚠️ No GPU — enable it in Runtime > Change runtime type"
!pip -q install -U transformers datasets
""")

md("## 1 · Project source code\nThe cells below write the project's `src/` package (identical to the local repo).")
code("import os\nos.makedirs('src', exist_ok=True)\nos.makedirs('data', exist_ok=True)")
for f in ["__init__.py", "config.py", "data.py", "metrics.py", "model.py", "models_rnn_lstm.py",
          "source_verifier.py", "train_transformer.py", "train_rnn_lstm.py", "train_baseline.py",
          "compare.py", "predict.py"]:
    code(f"%%writefile src/{f}\n" + (SRC / f).read_text(encoding="utf-8"))

md("## 2 · Dataset preparation & quick EDA")
code("""
!python -m src.data
""")
code("""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.data import get_splits
train, val, test = get_splits()
df = pd.concat([train, val, test])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x=df.label.map({0: 'Fake', 1: 'Real'}), ax=ax[0], palette=['#ff4d6d', '#00d4a6'])
ax[0].set_title('Class distribution'); ax[0].set_xlabel('')
df['words'] = df.content.str.split().str.len()
sns.histplot(data=df, x='words', hue=df.label.map({0: 'Fake', 1: 'Real'}), bins=80, ax=ax[1],
             palette=['#ff4d6d', '#00d4a6'], element='step')
ax[1].set_xlim(0, 1500); ax[1].set_title('Article length (words)')
plt.tight_layout(); plt.show()
print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
""")

md("## 3 · Baseline Models — TF-IDF, Simple RNN & BiLSTM")
code("!python -m src.train_baseline")
code("!python -m src.train_rnn_lstm --arch rnn --epochs 3")
code("!python -m src.train_rnn_lstm --arch lstm --epochs 3")

md("## 4 · Fine-tune BERT (`bert-base-uncased`)")
code("!python -m src.train_transformer --model bert --epochs 2 --max-len 256 --batch-size 16 --log-every 50")

md("## 5 · Fine-tune RoBERTa (`roberta-base`)")
code("!python -m src.train_transformer --model roberta --epochs 2 --max-len 256 --batch-size 16 --log-every 50")

md("## 6 · Comprehensive Model Comparison (TF-IDF vs Simple RNN vs BiLSTM vs BERT vs RoBERTa)")
code("""
!python -m src.compare
from IPython.display import Image, display, Markdown
display(Markdown(open('reports/comparison.md').read()))
display(Image('reports/figures/model_comparison.png'))
for n in ['tfidf_logreg', 'rnn', 'lstm', 'bert', 'roberta']:
    p = f'reports/figures/{n}_confusion_matrix.png'
    if os.path.exists(p): display(Image(p, width=380))
""")

md("## 7 · Source Analysis & Fact-Check Validation Test (English & Hindi)")
code("""
from src.predict import available_models, load_predictor
from src.source_verifier import analyze_source

models = available_models()
predictor = load_predictor(models['bert']) if 'bert' in models else load_predictor(next(iter(models.values())))

test_cases = [
    ("Scientists discover a new planet that can support human life.", "nasa.gov"),
    ("Government announces that humans can now live without food.", "fake-news.co"),
    ("ISRO launches new earth observation satellite successfully.", "pib.gov.in"),
    ("सरकारी आयोग का ऐलान: इंसानों को भोजन की जरूरत नहीं होगी!", "unverified-blog.com")
]

for text, src in test_cases:
    r = predictor.predict(text, explain=True)
    src_res = analyze_source(src, r.label, r.confidence)
    print(f"Text: {text}\\nSource: {src_res.domain} (Credibility: {src_res.credibility})")
    print(f"Prediction: {r.label} | Confidence: {r.confidence*100:.1f}% | Combined Trust Index: {src_res.combined_confidence*100:.1f}%\\n" + "-"*60)
""")

md("## 8 · Download trained models for local Streamlit app")
code("""
!zip -qr trained_models.zip models reports
from google.colab import files
files.download('trained_models.zip')
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}")

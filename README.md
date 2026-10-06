# 🛡️ TruthGuard: Multilingual & Explainable Fake News Detection Studio

<div align="center">

![TruthGuard Banner](assets/truthguard_logo.png)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**An advanced, multilingual, and explainable AI system engineered to classify news authenticity, attribute word-level neural influences, estimate prediction risk, and verify publisher reputation.**

[Key Features](#-key-features) • [Screenshots](#-interface--visual-tour) • [Architecture](#-model-architectures--benchmarks) • [Quickstart](#-installation--quickstart) • [Explainable AI](#-explainable-ai-xai)

</div>

---

## 📸 Interface & Visual Tour

### 1. News Detection Studio & Multi-Modal Input
> Clean, responsive interface featuring multi-modal inputs (Direct Text, Live News URL Scraper, PDF/Text Document upload), automatic language detection, and quick example pre-sets.

![TruthGuard Detector](docs/screenshots/truthguard_detector.png)

---

### 2. Prediction Verdict, Risk Meter & Explainable AI (XAI)
> Instant Fake/Real classification with 3D certainty sphere, risk meter (Low / Medium / High), softmax probability distributions, and word-level Integrated Gradients token attribution.

![Verdict and XAI Analysis](docs/screenshots/verdict_xai_analysis.png)

---

### 3. Model Benchmark & Empirical Leaderboard
> Comprehensive evaluation suite comparing Transformer (BERT), Recurrent Neural Networks (BiLSTM, Simple RNN), and TF-IDF baselines with confusion matrices and performance metrics.

![Model Comparison](docs/screenshots/model_comparison.png)

---

## 🌟 Key Features

- **🌐 Multilingual NLP Engine**:
  - Automatically identifies and analyzes **English**, **Hindi (हिन्दी 🇮🇳)**, and **Marathi (मराठी 🇮🇳)** text.
  - Multi-script tokenization and Devanagari feature extraction.

- **📄 Multi-Modal Document & Link Analysis**:
  - **Direct Text**: Analyze headlines, press releases, or complete articles.
  - **URL News Scraper**: Paste any news link to extract article title, body, and domain.
  - **PDF & TXT Document Upload**: Extract text from `.pdf` and `.txt` files with instant verification.

- **📈 Confidence & Uncertainty Risk Meter**:
  - 🟢 **LOW RISK**: High-confidence REAL news ($\ge 75\%$).
  - 🔴 **HIGH RISK**: High-confidence FAKE news ($\ge 75\%$).
  - 🟡 **UNCERTAIN / MEDIUM RISK**: Predictions with $50\% - 65\%$ confidence trigger an advisory:
    > *⚠️ Low confidence assessment — manual verification recommended.*

- **🧠 Explainable AI (XAI) Token Attribution**:
  - Token-level influence using **Integrated Gradients** (PyTorch) and linear feature attribution (TF-IDF).
  - Highlights words pushing towards **FAKE (Red)** vs **REAL (Green)** with exact attribution scores.

- **🔎 Publisher & Source Credibility Index**:
  - Cross-references news domains with accredited fact-checking agencies (PIB Fact Check, Reuters, BBC, AltNews, Snopes, Associated Press).
  - Evaluates domain age, WHOIS heuristic flags, and domain credibility scores.

- **📜 Prediction History & 📥 Downloadable Reports**:
  - Live session audit log tracking verdicts, confidence, and timestamps.
  - Export session logs to CSV.
  - Download formal `.txt` verification reports with XAI token attributions and academic disclaimers.

---

## 📊 Model Architectures & Benchmarks

All models were evaluated on a held-out test split of real-world annotated news datasets:

| Architecture | Model Family | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LogReg** | Scikit-Learn Pipeline | **98.50%** | **98.28%** | **98.71%** | **98.50%** | **0.9990** |
| **BERT** | HuggingFace Transformer | **96.50%** | **95.12%** | **97.80%** | **96.44%** | **0.9912** |
| **BiLSTM** | PyTorch Bidirectional LSTM | **93.00%** | **90.78%** | **94.12%** | **92.42%** | **0.9780** |
| **Simple RNN** | PyTorch Recurrent Net | **58.00%** | **55.20%** | **61.40%** | **58.14%** | **0.6200** |

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/tansawant/TruthGuard-Fake-News-Detector.git
cd TruthGuard-Fake-News-Detector
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧠 Explainable AI (XAI)

TruthGuard avoids "black box" machine learning by attributing neural weights back to individual input tokens:
- **Transformer & RNN Models**: Computes path integrals via **Integrated Gradients** across embedding spaces:
  $$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
- **Feature Weights**: Visualizes the top $k$ influential tokens in both positive and negative directions.

---

## 📁 Project Structure

```
truthguard-fake-news-detector/
│
├── app.py                      # Streamlit Multi-Page Web Application
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project documentation & screenshots
│
├── assets/                     # Branding logos and icons
│   └── truthguard_logo.png
│
├── docs/                       # Visual assets & screenshots
│   └── screenshots/
│       ├── truthguard_detector.png
│       ├── verdict_xai_analysis.png
│       └── model_comparison.png
│
├── src/                        # Core NLP & ML Modules
│   ├── config.py               # Path configurations & hyperparameter defaults
│   ├── dataset.py              # Data loading & preprocessing pipelines
│   ├── features.py             # PDF/URL extraction, risk estimation & language detector
│   ├── models.py               # PyTorch LSTM, RNN & Transformer network architectures
│   ├── predict.py              # Unified inference engine with @st.cache_resource
│   ├── source_verifier.py      # Fact-check domain index & credibility scoring
│   ├── train.py                # Model training scripts
│   └── xai.py                  # Integrated Gradients & token attribution routines
│
├── data/                       # Datasets
│   └── sample_news.csv         # Sample benchmark dataset (1,000 articles)
│
├── models/                     # Trained weights & metadata
│   ├── lstm/                   # BiLSTM model weights & vocab
│   ├── rnn/                    # Simple RNN model weights & vocab
│   └── tfidf_logreg.joblib     # TF-IDF + Logistic Regression pipeline
│
└── reports/                    # Quantitative evaluation metrics & confusion matrices
    ├── bert_metrics.json
    ├── lstm_metrics.json
    ├── rnn_metrics.json
    └── tfidf_logreg_metrics.json
```

---

## ⚖️ Academic Disclaimer

> **Notice**: TruthGuard provides probabilistic AI-assisted authenticity assessments based on linguistic patterns, statistical semantics, and domain registries. It does not replace journalistic verification or establish absolute factual truth. Users are encouraged to cross-reference sensitive or critical claims with accredited news agencies and primary sources.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

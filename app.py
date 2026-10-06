"""🛡️ TruthGuard — Multilingual & Explainable Fake News Detection System.

Full multi-page architecture with sidebar navigation:
- 🏠 Home (Project intro, supported models, capabilities)
- 🔍 News Detector (Multi-modal: Text, URL, PDF; Language detection; Risk/Uncertainty; XAI; Download report)
- 🔎 Source Verification (Domain credibility & fact-check registry)
- 📊 Model Comparison (BERT, BiLSTM, RNN, TF-IDF benchmarks, confusion matrix)
- 🧠 Explainable AI (Integrated Gradients deep-dive & token influence explorer)
- 📜 Prediction History (Session history log with CSV export)
- 🧩 How It Works (NLP pipeline architecture)
- ℹ️ About Project (Academic metadata, dataset, tech stack, disclaimers)
"""
from __future__ import annotations

import base64
import datetime
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import REPORTS_DIR
from src.predict import available_models, load_predictor
from src.source_verifier import TRUSTED_DOMAINS, UNTRUSTED_DOMAINS, analyze_source, extract_domain
from src.features import (
    assess_risk,
    detect_language,
    extract_claims,
    extract_text_from_pdf,
    extract_text_from_url,
    generate_assessment_report,
)

# ----------------------------------------------------------------------------- Page Config & Favicon
LOGO_PATH = Path(__file__).parent / "assets" / "truthguard_logo.png"
b64_logo = ""
if LOGO_PATH.exists():
    logo_img = Image.open(LOGO_PATH)
    with open(LOGO_PATH, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode("utf-8")
    st.set_page_config(page_title="TruthGuard AI · Fake News Detection",
                       page_icon=logo_img, layout="wide", initial_sidebar_state="expanded")
else:
    logo_img = None
    st.set_page_config(page_title="TruthGuard AI · Fake News Detection",
                       page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

if b64_logo:
    st.markdown(f"""
    <head>
        <link rel="icon" type="image/png" href="data:image/png;base64,{b64_logo}">
        <link rel="shortcut icon" href="data:image/png;base64,{b64_logo}">
    </head>
    """, unsafe_allow_html=True)

st.markdown("""
<script>
try {
    for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && (k.toLowerCase().includes('sidebar') || k.toLowerCase().includes('collapse'))) {
            localStorage.removeItem(k);
        }
    }
} catch(e) {}

function ensureSidebarAlwaysOpen() {
    try {
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        const isCollapsed = !sidebar || sidebar.getAttribute('aria-expanded') === 'false' || sidebar.getBoundingClientRect().width < 50;
        if (isCollapsed) {
            const btn = document.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                        document.querySelector('[data-testid="collapsedControl"] button') ||
                        document.querySelector('button[aria-label="Open sidebar"]') ||
                        document.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                        document.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                        document.querySelector('[data-testid="collapsedControl"]');
            if (btn) {
                btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
                btn.click();
            }
        }
    } catch(e) {}
}
ensureSidebarAlwaysOpen();
setTimeout(ensureSidebarAlwaysOpen, 100);
setTimeout(ensureSidebarAlwaysOpen, 300);
setTimeout(ensureSidebarAlwaysOpen, 700);
setTimeout(ensureSidebarAlwaysOpen, 1500);
</script>
""", unsafe_allow_html=True)

# Primary Colors
FAKE_C, REAL_C, ACC_C = "#dc2626", "#059669", "#4f46e5"

PRETTY = {
    "bert": "BERT Transformer",
    "roberta": "RoBERTa Transformer",
    "distilbert": "DistilBERT Transformer",
    "lstm": "BiLSTM Neural Net",
    "rnn": "Simple RNN Neural Net",
    "tfidf_logreg": "TF-IDF + LogReg Baseline"
}

# Multilingual UI Dictionary
I18N = {
    "en": {
        "title_prefix": "Fake",
        "title_highlight": "News Authenticity",
        "title_suffix": "Verifier",
        "subtitle": "AI-Powered Multilingual & Explainable News Verification System",
        "pill": "NLP · TRANSFORMERS · EXPLAINABLE AI · FACT-CHECK INDEX",
        "try_example": "Try an example article:",
        "input_placeholder": "Paste news headline or full article text here...",
        "source_placeholder": "Source URL or Publisher (e.g. reuters.com, pib.gov.in, fake-news.co)",
        "source_label": "Source / Publisher URL (Optional)",
        "check_btn": "ANALYZE NEWS AUTHENTICITY",
        "min_words_warn": "Please enter at least a full headline (3+ words).",
        "pred_label": "PREDICTION VERDICT",
        "confidence": "Model Confidence Score",
        "prob_heading": "Fake / Real Probability Distribution",
        "words_analyzed": "tokens analyzed",
        "truncated_warn": "text truncated to model max length",
        "short_text_warn": "⚠️ Short inputs give the model less context. Full news articles are recommended for higher accuracy.",
        "conf_disclaimer": "ℹ️ The confidence score represents model estimated probability, not absolute legal proof.",
        "xai_heading": "🧠 Explainable AI — Word Attribution Breakdown",
        "top_words_title": "Most Influential Words",
        "source_analysis_title": "🌐 Source & Publisher Reputation Analysis",
        "credibility": "Publisher Credibility",
        "domain_trust": "Domain Trust Score",
        "combined_trust": "Validated Credibility Index",
        "legend_fake": "pushes towards FAKE",
        "legend_real": "pushes towards REAL",
        "legend_hover": "Intensity = attribution strength · hover over words for exact scores",
        "all_models_title": "⚖️ All Models On This Article",
    },
    "hi": {
        "title_prefix": "फेक न्यूज़",
        "title_highlight": "प्रामाणिकता सत्यापन",
        "title_suffix": "प्रणाली",
        "subtitle": "एआई-संचालित बहुभाषी और व्याख्यात्मक समाचार प्रमाणिकता जांच",
        "pill": "एनएलपी · ट्रांसफॉर्मर · व्याख्यात्मक एआई · तथ्य जांच सूचकांक",
        "try_example": "उदाहरण समाचार चुनें:",
        "input_placeholder": "समाचार का शीर्षक या पूरा लेख यहाँ दर्ज करें...",
        "source_placeholder": "स्रोत यूआरएल या प्रकाशक (उदा. pib.gov.in, reuters.com, fake-news.co)",
        "source_label": "समाचार का स्रोत यूआरएल (वैकल्पिक)",
        "check_btn": "समाचार की प्रामाणिकता जांचें",
        "min_words_warn": "कृपया कम से कम एक पूरा शीर्षक (3+ शब्द) दर्ज करें।",
        "pred_label": "पूर्वानुमान का परिणाम",
        "confidence": "मॉडल आत्मविश्वास (Confidence Score)",
        "prob_heading": "फर्जी / असली संभावना वितरण",
        "words_analyzed": "शब्दों का विश्लेषण किया गया",
        "truncated_warn": "पाठ को मॉडल की अधिकतम लंबाई तक सीमित किया गया",
        "short_text_warn": "⚠️ छोटे वाक्यों से मॉडल को कम संदर्भ मिलता है। सटीक परिणाम के लिए पूरा समाचार दर्ज करें।",
        "conf_disclaimer": "ℹ️ कॉन्फिडेंस स्कोर मॉडल की अनुमानित संभावना है, अंतिम कानूनी प्रमाण नहीं।",
        "xai_heading": "🧠 व्याख्यात्मक एआई — शब्दों का प्रभाव विश्लेषण",
        "top_words_title": "सबसे प्रभावशाली शब्द",
        "source_analysis_title": "🌐 स्रोत और प्रकाशक प्रतिष्ठा विश्लेषण",
        "credibility": "प्रकाशक विश्वसनीयता",
        "domain_trust": "डोमेन ट्रस्ट स्कोर",
        "combined_trust": "सत्यापित प्रामाणिकता सूचकांक",
        "legend_fake": "फेक (FAKE) की तरफ ले जाता है",
        "legend_real": "असली (REAL) की तरफ ले जाता है",
        "legend_hover": "रंग की तीव्रता = प्रभाव की ताकत · शब्दों पर होवर करें",
        "all_models_title": "⚖️ इस समाचार पर सभी मॉडलों का परिणाम",
    },
    "mr": {
        "title_prefix": "खोटी बातमी",
        "title_highlight": "सत्यता पडताळणी",
        "title_suffix": "प्रणाली",
        "subtitle": "एआई-आधारित बहुभाषिक आणि स्पष्टीकरणात्मक बातम्यांची सत्यता पडताळणी प्रणाली",
        "pill": "एनएलपी · ट्रान्सफॉर्मर · स्पष्टीकरणात्मक एआई · तथ्य पडताळणी",
        "try_example": "उदाहरणाखालची बातमी निवडा:",
        "input_placeholder": "बातमीचे शीर्षक किंवा संपूर्ण मजकूर येथे प्रविष्ट करा...",
        "source_placeholder": "बातमीचा स्रोत किंवा संकेतस्थळ (उदा. reuters.com, pib.gov.in, fake-news.co)",
        "source_label": "बातमीचा स्रोत / URL (पर्यायी)",
        "check_btn": "बातमीची सत्यता तपासा",
        "min_words_warn": "कृपया किमान एक संपूर्ण शीर्षक (३+ शब्द) प्रविष्ट करा.",
        "pred_label": "भाकित निकाल (Prediction)",
        "confidence": "मॉडेल आत्मविश्वास (Confidence Score)",
        "prob_heading": "खोटी / खरी संभाव्यता वितरण",
        "words_analyzed": "शब्दांचे विश्लेषण केले",
        "truncated_warn": "मजकूर मॉडेलच्या कमाल मर्यादेपर्यंत मर्यादित केला",
        "short_text_warn": "⚠️ लहान वाक्यांमुळे मॉडेलला कमी संदर्भ मिळतो. अचूकतेसाठी संपूर्ण बातमी टाका.",
        "conf_disclaimer": "ℹ️ कॉन्फिडन्स स्कोर हे मॉडेलचे अंदाजित प्रमाण आहे, अंतिम पुरावा नाही.",
        "xai_heading": "🧠 स्पष्टीकरणात्मक एआई — शब्दांचे प्रभाव विश्लेषण",
        "top_words_title": "सर्वात प्रभावशाली शब्द",
        "source_analysis_title": "🌐 स्रोत आणि बातमी संकेतस्थळ विश्लेषण",
        "credibility": "प्रकाशक विश्वासार्हता",
        "domain_trust": "डोमेन ट्रस्ट स्कोर",
        "combined_trust": "पडताळणीकृत विश्वासार्हता निर्देशांक",
        "legend_fake": "खोट्या (FAKE) बातमीकडे झुकवते",
        "legend_real": "खऱ्या (REAL) बातमीकडे झुकवते",
        "legend_hover": "रंगाची तीव्रता = शब्दाचा प्रभाव · शब्दावर होव्हर करा",
        "all_models_title": "⚖️ या बातमीवर सर्व मॉडेलचे निकाल",
    }
}

MULTILINGUAL_EXAMPLES = {
    "en": {
        "🇬🇧 Policy Report": (
            "Germany's Merkel says coalition talks will resume next week. BERLIN - German Chancellor "
            "Angela Merkel said on Monday that exploratory talks with the Social Democrats on forming a "
            "new government would continue next week, after negotiators failed to reach agreement on "
            "migration and tax policy. A spokesman for the party said both sides had made progress on "
            "pensions and that a final decision would be put to members in January."),
        "🚨 Clickbait Claim": (
            "BREAKING: Obama Caught Hiding SHOCKING Secret From Americans - This Is UNBELIEVABLE! "
            "You won't believe what the former president has been hiding from the American people for "
            "years. Insiders say the liberal mainstream media is desperately covering it up, but the "
            "truth is finally coming out. Share this before it gets deleted!"),
        "🌌 Science Discovery": (
            "NASA's James Webb Space Telescope has discovered a faint galaxy cluster from the early universe. "
            "Astronomers confirmed that observations reveal water vapor and organic molecules in planetary atmospheres."),
    },
    "hi": {
        "🇮🇳 इसरो उपग्रह (Real)": (
            "भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) ने आज श्रीहरिकोटा से अपने नए पृथ्वी अवलोकन उपग्रह का सफल "
            "प्रक्षेपण किया। वैज्ञानिकों के अनुसार यह उपग्रह मौसम पूर्वानुमान और आपदा प्रबंधन में महत्वपूर्ण भूमिका निभाएगा।"),
        "🚨 फर्जी दावा (Fake)": (
            "सरकारी आयोग का बड़ा ऐलान: अब इंसानों को भोजन की आवश्यकता नहीं होगी, पानी पीने से ही मिलेगी पूर्ण ऊर्जा! "
            "मंत्रालय ने गुप्त रिपोर्ट में दावा किया है कि विशेष जल से सभी बीमारियां ठीक हो जाएंगी।"),
        "📰 योजना रिपोर्ट (Real)": (
            "केंद्र सरकार ने ग्रामीण युवाओं के लिए नई कौशल विकास योजना की शुरुआत की है। इस योजना के तहत "
            "लाखों युवाओं को रोजगार के लिए नि:शुल्क डिजिटल प्रशिक्षण दिया जाएगा।"),
    },
    "mr": {
        "🇮🇳 इस्रो मोहीम (Real)": (
            "भारतीय अवकाश संशोधन संस्थेने (ISRO) श्रीहरिकोटा येथून नवीन उपग्रहाचे यशस्वी प्रक्षेपण केले. "
            "हा उपग्रह हवामान अंदाज आणि नैसर्गिक आपत्ती व्यवस्थापनात महत्त्वाची भूमिका बजावेल."),
        "🚨 बनावट दावा (Fake)": (
            "आरोग्य मंत्रालयाची मोठी घोषणा: केवळ उकळलेले हळदीचे पाणी प्यायल्याने सर्व आजार कायमचे बरे होतात! "
            "मंत्रालयाने गुप्त अहवालात दावा केला आहे की नागरिकांना आता औषधांची गरज भासणार नाही."),
        "📰 शेती योजना (Real)": (
            "राज्य सरकारने शेतकऱ्यांसाठी नवीन सौर कृषी पंप योजना जाहीर केली आहे. या योजनेअंतर्गत "
            "शेतकऱ्यांना ९० टक्के अनुदानावर सौर पंप उपलब्ध करून दिले जातील."),
    }
}

# ----------------------------------------------------------------------------- Clean Modern Aesthetic CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-main: #f8fafc;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #334155;
    --text-muted: #64748b;
    --accent-indigo: #4f46e5;
    --accent-blue: #0284c7;
    --fake-red: #dc2626;
    --fake-bg: #fef2f2;
    --real-green: #059669;
    --real-bg: #ecfdf5;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    color: #0f172a !important;
    z-index: 9999 !important;
}

header[data-testid="stHeader"] [data-testid="stToolbar"], #MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
}

/* Sidebar: Permanently Docked and Visible Directly at Left */
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
    margin-left: 0 !important;
    min-width: 335px !important;
    max-width: 360px !important;
    width: 345px !important;
    opacity: 1 !important;
    background: #ffffff !important;
    border-right: 1.5px solid #e2e8f0 !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.03) !important;
}

/* Hide collapse controls so the sidebar never vanishes */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(79, 70, 229, 0.06), transparent 70%),
        radial-gradient(1000px 500px at 100% 0%, rgba(5, 150, 105, 0.04), transparent 70%),
        #f8fafc;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1250px;
}

/* Sidebar Styles */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e2e8f0 !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.03);
}

/* Nav Radio in Sidebar */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 3px;
    transition: all 0.18s ease;
    font-weight: 600;
    color: #334155;
    cursor: pointer;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: #f1f5f9;
    color: #4f46e5;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[aria-checked="true"] {
    background: #eef2ff !important;
    color: #3730a3 !important;
    font-weight: 700;
}

/* Top Navbar */
.top-nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 0.85rem 1.4rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04);
    margin-bottom: 1.4rem;
}

.brand-group {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-group img {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12);
}

.brand-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #0f172a;
    line-height: 1.15;
}

.brand-sub {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 600;
}

.nav-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.95rem;
    border-radius: 999px;
    font-size: 0.82rem;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    font-weight: 700;
}

.nav-status-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #16a34a;
    box-shadow: 0 0 8px #16a34a;
}

.nav-action-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 0.42rem 0.85rem;
    font-size: 0.84rem;
    font-weight: 700;
    color: #1e293b;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    font-family: 'Inter', sans-serif;
}

.nav-action-btn:hover {
    border-color: #4f46e5;
    color: #4f46e5;
    background: #f8fafc;
    transform: translateY(-1px);
}

/* Hero Banner */
.hero-banner {
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 55%, #f0fdf4 100%);
    border: 1px solid #cbd5e1;
    box-shadow: 0 12px 35px -12px rgba(79, 70, 229, 0.12);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-banner:after {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(79, 70, 229, 0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.hero-banner h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2.7rem;
    color: #0f172a;
    line-height: 1.15;
    margin: 0.5rem 0 0.55rem 0;
    letter-spacing: -0.025em;
}

.grad-text {
    background: linear-gradient(90deg, #4f46e5 0%, #0284c7 45%, #059669 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-banner p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    max-width: 820px;
    margin: 0;
    line-height: 1.55;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    background: #e0e7ff;
    border: 1px solid #c7d2fe;
    color: #3730a3;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #059669;
    box-shadow: 0 0 8px #059669;
}

/* Cards */
.card-light {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1.2rem;
}

.card-light:hover {
    box-shadow: 0 8px 25px -4px rgba(0, 0, 0, 0.08);
}

.card-light h3 {
    font-family: 'Outfit', sans-serif;
    color: #0f172a;
    font-size: 1.18rem;
    font-weight: 700;
    margin-top: 0;
}

/* Verdict Cards with 3D Sphere */
.verdict-card {
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    position: relative;
    border: 2px solid #e2e8f0;
    box-shadow: 0 10px 30px -5px rgba(0,0,0,0.06);
}

.verdict-card.fake {
    background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
    border-color: #fca5a5;
}

.verdict-card.real {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    border-color: #6ee7b7;
}

.verdict-card.uncertain {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-color: #fcd34d;
}

.verdict-card .lbl {
    font-size: 0.82rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 800;
    color: var(--text-muted);
}

.verdict-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 0.4rem 0 0.5rem 0;
}

.sphere-3d {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    flex-shrink: 0;
}

.sphere-3d.real {
    background: radial-gradient(circle at 35% 35%, #a7f3d0 0%, #10b981 50%, #047857 100%);
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.45);
}

.sphere-3d.fake {
    background: radial-gradient(circle at 35% 35%, #fecaca 0%, #ef4444 50%, #b91c1c 100%);
    box-shadow: 0 6px 18px rgba(239, 68, 68, 0.45);
}

.sphere-3d.uncertain {
    background: radial-gradient(circle at 35% 35%, #fef08a 0%, #eab308 50%, #a16207 100%);
    box-shadow: 0 6px 18px rgba(234, 179, 8, 0.45);
}

.verdict-card .big {
    font-family: 'Outfit', sans-serif;
    font-size: 3.3rem;
    font-weight: 800;
    line-height: 1;
}

.verdict-card.fake .big { color: #dc2626; }
.verdict-card.real .big { color: #059669; }
.verdict-card.uncertain .big { color: #d97706; }

.verdict-card .conf {
    font-size: 1.12rem;
    color: #1e293b;
    font-weight: 600;
}

/* Probability Bars */
.pbar-light { margin: 0.95rem 0; }
.pbar-light .top { display: flex; justify-content: space-between; font-weight: 700; font-size: 0.95rem; color: #1e293b; margin-bottom: 0.35rem; }
.pbar-light .track { height: 15px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }
.pbar-light .fill { height: 100%; border-radius: 999px; width: 0; transition: width 0.9s cubic-bezier(0.2, 0.9, 0.3, 1); }
.pbar-light.fake .fill { background: linear-gradient(90deg, #f87171, #dc2626); box-shadow: 0 0 12px rgba(220, 38, 38, 0.3); }
.pbar-light.real .fill { background: linear-gradient(90deg, #34d399, #059669); box-shadow: 0 0 12px rgba(5, 150, 105, 0.3); }

/* Highlighted Text */
.hl-light { line-height: 2.15; font-size: 1.02rem; color: #0f172a; }
.hl-light span.w { padding: 0.2rem 0.38rem; border-radius: 6px; display: inline; font-weight: 500; transition: transform 0.15s ease; }
.hl-light span.cut { color: #94a3b8; }

/* Risk & Warning Box */
.risk-meter-box {
    border-radius: 14px;
    padding: 0.9rem 1.15rem;
    font-size: 0.92rem;
    margin-top: 0.8rem;
    font-weight: 600;
    line-height: 1.5;
}

.warn-box {
    border-left: 4px solid #f59e0b;
    background: #fffbeb;
    border-radius: 12px;
    padding: 0.9rem 1.15rem;
    color: #92400e;
    font-size: 0.9rem;
    margin-top: 1rem;
    font-weight: 600;
}

/* Input Fields & Buttons */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 16px !important;
    color: #0f172a !important;
    font-size: 1.02rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3.5px rgba(79, 70, 229, 0.18) !important;
}

.stTextInput input {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 14px !important;
    color: #0f172a !important;
}

.stButton>button {
    border-radius: 14px;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #1e293b;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    border-color: #4f46e5;
    background: #f8fafc;
    color: #4f46e5;
    transform: translateY(-1px);
}

.stButton>button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #4f46e5 0%, #0284c7 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    padding: 0.85rem 1.4rem !important;
    letter-spacing: 0.02em;
    font-size: 1.05rem !important;
    box-shadow: 0 8px 24px -6px rgba(79, 70, 229, 0.45) !important;
    border-radius: 14px !important;
}

.stButton>button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -6px rgba(79, 70, 229, 0.55) !important;
}

/* Metric Info Card in Sidebar */
.sidebar-metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem;
    margin-top: 0.6rem;
}

.sidebar-metric-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #0f172a;
}

.sidebar-metric-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- Helpers & Cache
@st.cache_resource(show_spinner="Loading model...")
def get_predictor(path: str):
    return load_predictor(Path(path))


@st.cache_data(ttl=60)
def load_reports() -> dict:
    out = {}
    for f in REPORTS_DIR.glob("*_metrics.json"):
        m = json.loads(f.read_text())
        out[m["model"]] = m
    return out


def plot_layout(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color="#0f172a"),
                      legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    return fig


def verdict_html(pred, lang_dict, risk_info) -> str:
    status_label, risk_level, risk_col, risk_desc = risk_info
    cls = status_label.lower()
    bars = "".join(
        f'<div class="pbar-light {k.lower()}"><div class="top"><span>{k.title()}</span><span>{v * 100:.1f}%</span></div>'
        f'<div class="track"><div class="fill" style="width:{max(v * 100, 2):.1f}%"></div></div></div>'
        for k, v in pred.probs.items())
    
    return (f'<div class="verdict-card {cls}">'
            f'<div class="lbl">{lang_dict["pred_label"]}</div>'
            f'<div class="verdict-row">'
            f'<div class="sphere-3d {cls}"></div>'
            f'<div class="big">{status_label}</div>'
            f'</div>'
            f'<div class="conf">{lang_dict["confidence"]}: <b>{pred.confidence * 100:.1f}%</b></div>'
            f'<div class="risk-meter-box" style="background:rgba(255,255,255,0.7); border:1px solid {risk_col}; color:{risk_col};">'
            f'<b>Risk Level: {risk_level}</b><br><span style="font-size:0.85rem; color:#475569;">{risk_desc}</span>'
            f'</div>'
            f'</div>'
            f'<div class="card-light" style="margin-top:1.2rem"><h3>{lang_dict["prob_heading"]}</h3>{bars}'
            f'<div style="font-size:0.85rem; color:#64748b; margin-top:0.4rem">'
            f'Softmax · {pred.n_tokens} {lang_dict["words_analyzed"]}'
            f'{" (" + lang_dict["truncated_warn"] + ")" if pred.truncated else ""}</div></div>')


def highlight_html(words, lang_dict) -> str:
    parts = []
    for w, s in words:
        esc = html.escape(w)
        if s is None:
            parts.append(f'<span class="w cut">{esc}</span>')
        elif abs(s) < 0.06:
            parts.append(f'<span class="w">{esc}</span>')
        else:
            bg = "#fee2e2" if s > 0 else "#d1fae5"
            border = "#fca5a5" if s > 0 else "#6ee7b7"
            color = "#991b1b" if s > 0 else "#065f46"
            tip = f"{'→ FAKE' if s > 0 else '→ REAL'} ({s:+.2f})"
            parts.append(f'<span class="w" title="{tip}" style="background:{bg}; border:1px solid {border}; color:{color}">{esc}</span>')
    return (f'<div class="card-light"><h3>{lang_dict["xai_heading"]}</h3><div class="hl-light">' + " ".join(parts) +
            f'</div><div style="display:flex; gap:1.2rem; font-size:0.85rem; color:#64748b; margin-top:0.8rem; flex-wrap:wrap;">'
            f'<span><b style="display:inline-block;width:12px;height:12px;background:#fee2e2;border:1px solid #fca5a5;border-radius:3px;margin-right:4px;"></b>{lang_dict["legend_fake"]}</span>'
            f'<span><b style="display:inline-block;width:12px;height:12px;background:#d1fae5;border:1px solid #6ee7b7;border-radius:3px;margin-right:4px;"></b>{lang_dict["legend_real"]}</span>'
            f'<span style="margin-left:auto">{lang_dict["legend_hover"]}</span></div></div>')


def top_words_fig(words, title, k=8):
    agg = defaultdict(float)
    for w, s in words:
        if s is None:
            continue
        key = re.sub(r"[^\w'$%-]", "", w).strip("'").lower() or w
        agg[key] += s
    items = sorted(agg.items(), key=lambda kv: kv[1])
    real = [kv for kv in items if kv[1] < 0][:k]
    fake = [kv for kv in reversed(items) if kv[1] > 0][:k]
    data = sorted(real + fake, key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in data], y=[w for w, _ in data], orientation="h",
                           marker_color=[FAKE_C if v > 0 else REAL_C for _, v in data],
                           hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(family="Outfit", size=15, color="#0f172a")))
    fig.update_xaxes(title="← REAL      attribution      FAKE →")
    return plot_layout(fig, h=max(260, 26 * len(data) + 90))


def gauge_fig(pred, title, risk_level):
    color = FAKE_C if pred.label == "FAKE" else REAL_C
    if risk_level == "MEDIUM":
        color = "#d97706"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pred.confidence * 100, number=dict(suffix="%", font=dict(family="Outfit", size=38, color="#0f172a")),
        title=dict(text=title, font=dict(size=14, color="#64748b")),
        gauge=dict(axis=dict(range=[50, 100], tickcolor="#cbd5e1"), bar=dict(color=color, thickness=0.32),
                   bgcolor="#f1f5f9", borderwidth=0,
                   steps=[dict(range=[50, 70], color="#f8fafc"),
                          dict(range=[70, 90], color="#f1f5f9"),
                          dict(range=[90, 100], color="#e2e8f0")])))
    return plot_layout(fig, h=230)


def cm_fig(cm, title):
    tot = [max(sum(r), 1) for r in cm]
    text = [[f"{cm[i][j]:,}<br>({cm[i][j] / tot[i]:.1%})" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=[[cm[i][j] / tot[i] for j in range(2)] for i in range(2)],
                               x=["Pred Fake", "Pred Real"], y=["Actual Fake", "Actual Real"],
                               text=text, texttemplate="%{text}", textfont=dict(size=15, family="Outfit"),
                               colorscale=[[0, "#f8fafc"], [0.5, "#818cf8"], [1, "#4f46e5"]],
                               zmin=0, zmax=1, showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text=title, font=dict(family="Outfit", size=15, color="#0f172a")))
    return plot_layout(fig, h=300)


def set_example(txt):
    st.session_state["news_text"] = txt

def set_source(src):
    st.session_state["source_url"] = src


# ----------------------------------------------------------------------------- Session State Initialization
if "history" not in st.session_state:
    st.session_state["history"] = []

if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

if "source_url" not in st.session_state:
    st.session_state["source_url"] = ""

models = available_models()
reports = load_reports()

# ----------------------------------------------------------------------------- Sidebar Navigation & Controls
with st.sidebar:
    if b64_logo:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.95rem; padding-bottom:0.8rem; border-bottom:1.5px solid #f1f5f9;">
            <img src="data:image/png;base64,{b64_logo}" width="42" height="42" style="border-radius:10px; box-shadow:0 4px 10px rgba(79,70,229,0.15); object-fit:cover;">
            <div>
                <div style="font-family:'Outfit'; font-weight:800; font-size:1.22rem; color:#0f172a; line-height:1.1;">🛡️ TruthGuard</div>
                <div style="font-size:0.74rem; color:#64748b; font-weight:600;">Multilingual Fake News Detection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom:0.95rem; padding-bottom:0.8rem; border-bottom:1.5px solid #f1f5f9;">
            <div style="font-family:'Outfit'; font-weight:800; font-size:1.25rem; color:#0f172a; line-height:1.1;">🛡️ TruthGuard</div>
            <div style="font-size:0.75rem; color:#64748b; font-weight:600;">Multilingual Fake News Detection</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <script>
    try {
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k && (k.toLowerCase().includes('sidebar') || k.toLowerCase().includes('collapse'))) {
                localStorage.removeItem(k);
            }
        }
    } catch(e) {}

    function autoExpandSidebar() {
        try {
            const sidebar = document.querySelector('section[data-testid="stSidebar"]');
            const isCollapsed = !sidebar || sidebar.getAttribute('aria-expanded') === 'false' || sidebar.getBoundingClientRect().width < 50;
            if (isCollapsed) {
                const btn = document.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                            document.querySelector('[data-testid="collapsedControl"] button') ||
                            document.querySelector('button[aria-label="Open sidebar"]') ||
                            document.querySelector('[data-testid="stSidebarCollapseButton"] button');
                if (btn) {
                    btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
                    btn.click();
                }
            }
        } catch(e) {}
    }
    autoExpandSidebar();
    setTimeout(autoExpandSidebar, 50);
    setTimeout(autoExpandSidebar, 200);
    setTimeout(autoExpandSidebar, 600);
    </script>
    """, unsafe_allow_html=True)

    # 🌐 Language & Model Settings Section (Matching exact user screenshot)
    st.markdown("""
    <div style="font-family:'Outfit', sans-serif; font-size:1.18rem; font-weight:700; color:#0f172a; margin-bottom:0.7rem; display:flex; align-items:center; gap:8px;">
        <span style="font-size:1.3rem;">🌐</span> Language & Model Settings
    </div>
    """, unsafe_allow_html=True)
    
    lang = st.selectbox(
        "Select Application Language",
        options=["en", "hi", "mr"],
        format_func=lambda l: {"en": "GB English (Primary)", "hi": "IN Hindi (हिंदी)", "mr": "IN Marathi (मराठी)"}[l],
        key="lang_sel"
    )
    L = I18N[lang]

    if models:
        names = list(models.keys())
        default_idx = names.index("lstm") if "lstm" in names else (names.index("bert") if "bert" in names else 0)
        model_name = st.selectbox(
            "Select Model Architecture",
            names,
            index=default_idx,
            format_func=lambda n: PRETTY.get(n, n),
            key="model_sel"
        )

        explain = st.toggle("Explain prediction (XAI)", value=True,
                            help="Integrated Gradients token attribution for Transformer / RNN; feature weights for TF-IDF.")
        
        ig_steps = st.slider("Explanation precision (IG steps)", 8, 64, 24, 8,
                             disabled=not explain or models[model_name].suffix == ".joblib")
        
        compare_all = st.toggle("Run all models side-by-side", value=len(models) > 1)

        meta = get_predictor(str(models[model_name])).meta
        tm = meta.get("test_metrics")

        base_arch = meta.get("base_model", "Deep Neural Net")
        if "lstm" in model_name.lower():
            base_arch = "PyTorch LSTM"
        elif "bert" in model_name.lower():
            base_arch = "HuggingFace Transformers"
        elif "rnn" in model_name.lower():
            base_arch = "PyTorch Simple RNN"
        elif "tfidf" in model_name.lower():
            base_arch = "Scikit-Learn Pipeline"

        st.markdown(f"""
        <div class="sidebar-metric-card" style="margin-top:0.8rem;">
            <div class="sidebar-metric-title">{PRETTY.get(model_name, model_name)}</div>
            <div class="sidebar-metric-sub">{base_arch}</div>
        """, unsafe_allow_html=True)
        if tm:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{tm['accuracy'] * 100:.1f}%")
            c2.metric("F1 (Fake)", f"{tm['f1'] * 100:.1f}%")
            if meta.get("train_size"):
                st.caption(f"Trained on {meta['train_size']:,} samples")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='border-top:1.5px solid #e2e8f0; margin:1.2rem 0 0.8rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>🛡️ TruthGuard Navigation</div>", unsafe_allow_html=True)

    PAGES = [
        "🏠 Home",
        "🔍 News Detector",
        "🔎 Source Verification",
        "📊 Model Comparison",
        "🧠 Explainable AI",
        "📜 Prediction History",
        "🧩 How It Works",
        "ℹ️ About Project",
    ]

    selected_page = st.radio(
        "Navigation Menu",
        options=PAGES,
        index=1,  # Default to News Detector
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.caption("ℹ️ Probabilistic NLP assessment. Cross-check with accredited news agencies.")


# ----------------------------------------------------------------------------- Top Navigation Header Bar
lang_names = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
lang_display = lang_names.get(lang, "English")

if b64_logo:
    st.markdown(f"""
    <div class="top-nav-bar">
        <div class="brand-group">
            <img src="data:image/png;base64,{b64_logo}" alt="TruthGuard Logo">
            <div>
                <div class="brand-title">TruthGuard AI</div>
                <div class="brand-sub">Multilingual News Intelligence & Authenticity Engine</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <div class="nav-status-pill">
                <span class="dot"></span>
                <span>Active Model: {PRETTY.get(model_name, model_name)} · {lang_display}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================= PAGE 1: 🏠 HOME
if selected_page == "🏠 Home":
    st.markdown(f"""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>NLP · TRANSFORMERS · EXPLAINABLE AI · FACT-CHECK REGISTRY</span>
      <h1>TruthGuard <span class="grad-text">AI Verification Studio</span></h1>
      <p>An advanced, explainable, multilingual machine learning and deep learning system engineered to classify news as <b>REAL</b> or <b>FAKE</b>, analyze publisher credibility, and interpret neural token attributions.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:1.2rem; margin-bottom:1.5rem;">
        <div class="card-light">
            <h3>🧠 Deep Learning Architectures</h3>
            <p style="color:#475569; font-size:0.92rem; line-height:1.6;">
                Trained on contextual Transformers (<b>BERT</b>, <b>RoBERTa</b>), Bidirectional LSTM (<b>BiLSTM</b>), Simple RNN, and TF-IDF baselines.
            </p>
        </div>
        <div class="card-light">
            <h3>🌐 Multilingual NLP</h3>
            <p style="color:#475569; font-size:0.92rem; line-height:1.6;">
                Native support and script analysis for <b>English (Primary)</b>, <b>Hindi (हिंदी)</b>, and <b>Marathi (मराठी)</b> articles.
            </p>
        </div>
        <div class="card-light">
            <h3>🔬 Explainable AI (XAI)</h3>
            <p style="color:#475569; font-size:0.92rem; line-height:1.6;">
                Uses <b>Integrated Gradients</b> to attribute model confidence back to token embeddings, highlighting influential words.
            </p>
        </div>
        <div class="card-light">
            <h3>🔎 Fact-Check Domain Index</h3>
            <p style="color:#475569; font-size:0.92rem; line-height:1.6;">
                Cross-references publisher domains against accredited fact-checking agencies (PIB Fact Check, Reuters, BBC, AltNews, Snopes).
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Supported Models & Empirical Benchmarks")
    if reports:
        b_rows = []
        for k in ["tfidf_logreg", "bert", "lstm", "rnn"]:
            if k in reports:
                r = reports[k]
                b_rows.append({
                    "Architecture": PRETTY.get(k, k),
                    "Accuracy": f"{r['accuracy'] * 100:.2f}%",
                    "Precision (Fake)": f"{r['precision'] * 100:.2f}%",
                    "Recall (Fake)": f"{r['recall'] * 100:.2f}%",
                    "F1 Score": f"{r['f1'] * 100:.2f}%",
                    "ROC-AUC": f"{r.get('roc_auc', 0):.4f}",
                    "Test Samples": f"{r['n_test']:,}"
                })
        st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)


# ============================================================================= PAGE 2: 🔍 NEWS DETECTOR (DEFAULT)
elif selected_page == "🔍 News Detector":
    st.markdown(f"""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>{L["pill"]}</span>
      <h1>{L["title_prefix"]} <span class="grad-text">{L["title_highlight"]}</span> {L["title_suffix"]}</h1>
      <p>{L["subtitle"]}</p>
    </div>""", unsafe_allow_html=True)

    if not models:
        st.error("No trained models found in the models directory.")
        st.stop()

    # Input Modalities: Direct Text, URL, PDF
    input_mode = st.radio("Choose Input Mode", ["✍️ Direct Text", "🔗 URL Article Scraper", "📄 PDF / Text Upload"],
                          horizontal=True, label_visibility="collapsed")

    # Example Buttons Bar (for Direct Text)
    if input_mode == "✍️ Direct Text":
        st.markdown(f"<div style='color:#64748b; font-weight:600; margin-bottom:0.4rem; font-size:0.92rem'>{L['try_example']}</div>", unsafe_allow_html=True)
        ex_dict = MULTILINGUAL_EXAMPLES[lang]
        cols = st.columns(len(ex_dict))
        for i, (c, (k, v)) in enumerate(zip(cols, ex_dict.items())):
            c.button(k, key=f"ex_{lang}_{i}", on_click=set_example, args=(v,), use_container_width=True)

    elif input_mode == "🔗 URL Article Scraper":
        st.markdown("<div class='card-light'><b>Paste News Article URL</b>", unsafe_allow_html=True)
        u_col1, u_col2 = st.columns([3, 1])
        with u_col1:
            input_url = st.text_input("URL", placeholder="https://www.reuters.com/world/... or https://pib.gov.in/...", label_visibility="collapsed")
        with u_col2:
            fetch_btn = st.button("📥 Fetch & Extract Article", use_container_width=True)
        if fetch_btn and input_url:
            with st.spinner("Extracting content from URL..."):
                u_title, u_body, u_dom, u_err = extract_text_from_url(input_url)
            if u_err:
                st.error(u_err)
            else:
                st.session_state["news_text"] = u_body
                st.session_state["source_url"] = u_dom
                st.success(f"Extracted article from **{u_dom}**! Title: *{u_title}*")
        st.markdown("</div>", unsafe_allow_html=True)

    elif input_mode == "📄 PDF / Text Upload":
        st.markdown("<div class='card-light'><b>Upload News PDF or Text File</b>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                pdf_text = extract_text_from_pdf(uploaded_file.read())
                if pdf_text:
                    st.session_state["news_text"] = pdf_text
                    st.success(f"Extracted {len(pdf_text.split())} words from **{uploaded_file.name}**")
            else:
                txt_content = uploaded_file.read().decode("utf-8", errors="ignore")
                st.session_state["news_text"] = txt_content
                st.success(f"Extracted {len(txt_content.split())} words from **{uploaded_file.name}**")
        st.markdown("</div>", unsafe_allow_html=True)

    # Main Inputs
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    c_txt, c_src = st.columns([3, 1], gap="medium")
    with c_txt:
        text = st.text_area("News Text Input", key="news_text", height=185,
                            placeholder=L["input_placeholder"], label_visibility="collapsed")
    with c_src:
        source_url = st.text_input(L["source_label"], key="source_url",
                                   placeholder=L["source_placeholder"],
                                   help="Enter source domain or publisher URL to validate publisher reputation")
        st.markdown("<div style='font-size:0.75rem; color:#64748b; font-weight:600; margin-top:0.4rem;'>Quick Fill Domain:</div>", unsafe_allow_html=True)
        qs1, qs2, qs3 = st.columns(3)
        qs1.button("reuters", on_click=set_source, args=("reuters.com",), use_container_width=True, key="qs_reuters")
        qs2.button("pib.gov", on_click=set_source, args=("pib.gov.in",), use_container_width=True, key="qs_pib")
        qs3.button("fake.co", on_click=set_source, args=("fake-news.co",), use_container_width=True, key="qs_fake")

    # Language auto-detection & Claim extraction option
    detected_lang_code, detected_lang_label = detect_language(text)
    wc = len(text.split())

    opt_col1, opt_col2 = st.columns([2, 2])
    with opt_col1:
        st.markdown(f"<div style='font-size:0.85rem; color:#475569; padding-top:0.3rem;'>🌐 <b>Detected Language:</b> <span class='badge-pill' style='font-size:0.75rem; padding:0.15rem 0.55rem;'>{detected_lang_label}</span> · <b>{wc} words</b></div>", unsafe_allow_html=True)
    with opt_col2:
        extract_claims_toggle = st.checkbox("Granular Claim Extraction (Verify each sentence)", value=False)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    go_btn = st.button(L["check_btn"], type="primary", use_container_width=True, key="check_btn")

    if go_btn and wc < 3:
        st.warning(L["min_words_warn"])
    elif go_btn:
        predictor = get_predictor(str(models[model_name]))
        with st.spinner(f"{PRETTY.get(model_name, model_name)} analyzing authenticity..."):
            pred = predictor.predict(text, explain=explain, ig_steps=ig_steps)
            risk_info = assess_risk(pred.label, pred.confidence)
            src_analysis = analyze_source(source_url, pred.label, pred.confidence)
            
            # Log to session history
            now_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state["history"].append({
                "time": now_time,
                "text": text[:90] + ("..." if len(text) > 90 else ""),
                "full_text": text,
                "label": risk_info[0],
                "confidence": f"{pred.confidence * 100:.1f}%",
                "risk": risk_info[1],
                "model": PRETTY.get(model_name, model_name),
                "source": src_analysis.domain if src_analysis.domain != "Not Provided" else "None",
            })
            
            st.session_state["last"] = {
                "pred": pred,
                "model": model_name,
                "text": text,
                "source": source_url,
                "risk_info": risk_info,
                "src_analysis": src_analysis,
                "detected_lang": detected_lang_label
            }

    last = st.session_state.get("last")
    if last and last["text"] == text:
        pred = last["pred"]
        risk_info = last["risk_info"]
        src_analysis = last["src_analysis"]

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.markdown(verdict_html(pred, L, risk_info), unsafe_allow_html=True)
            if src_analysis.domain != "Not Provided":
                badge_col = "#059669" if src_analysis.credibility in ("High", "Medium-High") else "#dc2626"
                bg_col = "#ecfdf5" if src_analysis.credibility in ("High", "Medium-High") else "#fef2f2"
                st.markdown(f"""
                <div class="card-light" style="margin-top:1.2rem; border-color:{badge_col}; background:{bg_col}">
                    <h3 style="color:{badge_col}">{L["source_analysis_title"]}</h3>
                    <div style="font-weight:700; font-size:1.15rem; color:{badge_col}">{src_analysis.source_name} ({src_analysis.domain})</div>
                    <div style="font-size:0.92rem; color:#475569; margin-top:0.25rem">
                        <b>{L["credibility"]}:</b> {src_analysis.credibility} | <b>{L["domain_trust"]}:</b> {src_analysis.trust_score*100:.0f}%
                    </div>
                    <div style="font-size:0.92rem; margin-top:0.45rem; color:#1e293b; line-height:1.5">{src_analysis.description}</div>
                    <div style="font-size:0.92rem; margin-top:0.6rem; font-weight:700; color:#4f46e5">
                        {L["combined_trust"]}: {src_analysis.combined_confidence*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with right:
            st.plotly_chart(gauge_fig(pred, L["confidence"], risk_info[1]), use_container_width=True, config={"displayModeBar": False})
            if wc < 25:
                st.markdown(f"<div class='warn-box'>{L['short_text_warn']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='warn-box'>{L['conf_disclaimer']}</div>", unsafe_allow_html=True)

        # Granular Claims Breakdown
        if extract_claims_toggle:
            claims_list = extract_claims(text)
            if claims_list:
                st.markdown("<div style='height:1rem'></div><h3 style='font-family:Outfit; color:#0f172a'>📰 Granular Claim-by-Claim Breakdown</h3>", unsafe_allow_html=True)
                for idx, clm in enumerate(claims_list, 1):
                    c_res = get_predictor(str(models[model_name])).predict(clm, explain=False)
                    c_col = FAKE_C if c_res.label == "FAKE" else REAL_C
                    st.markdown(f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-left:4px solid {c_col}; border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.6rem; display:flex; justify-content:space-between; align-items:center;">
                        <div><b>Claim {idx}:</b> <i>"{clm}"</i></div>
                        <div><span style="font-weight:800; color:{c_col}; font-size:1.05rem;">{c_res.label}</span> <span style="font-size:0.85rem; color:#64748b;">({c_res.confidence*100:.1f}%)</span></div>
                    </div>
                    """, unsafe_allow_html=True)

        if pred.words:
            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
            h1, h2 = st.columns([1.6, 1], gap="large")
            h1.markdown(highlight_html(pred.words, L), unsafe_allow_html=True)
            h2.plotly_chart(top_words_fig(pred.words, L["top_words_title"]), use_container_width=True, config={"displayModeBar": False})

        if compare_all and len(models) > 1:
            st.markdown(f"<div style='height:1.4rem'></div><h3 style='font-family:Outfit; font-size:1.4rem; font-weight:800; color:#0f172a'>{L['all_models_title']}</h3>", unsafe_allow_html=True)
            mcols = st.columns(len(models))
            for c, (n, pth) in zip(mcols, models.items()):
                r = get_predictor(str(pth)).predict(text)
                col = FAKE_C if r.label == "FAKE" else REAL_C
                is_active = (n == model_name)
                border_style = f"border: 2px solid {ACC_C}; box-shadow: 0 4px 18px rgba(79, 70, 229, 0.12);" if is_active else "border: 1px solid #e2e8f0;"
                c.markdown(f"""
                <div class="card-light" style="text-align:center; {border_style}">
                    <div style="font-size:0.85rem; color:#64748b; font-weight:700">{PRETTY.get(n, n)}</div>
                    <div style="font-family:'Outfit'; font-size:2rem; font-weight:800; color:{col}; margin:0.35rem 0;">{r.label}</div>
                    <div style="font-weight:700; color:#0f172a; font-size:1.05rem;">{r.confidence * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        # Download Assessment Report
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        report_text = generate_assessment_report(
            news_text=text,
            pred_label=risk_info[0],
            confidence=pred.confidence,
            model_name=PRETTY.get(model_name, model_name),
            risk_level=risk_info[1],
            risk_desc=risk_info[3],
            source_domain=src_analysis.domain,
            source_trust_score=src_analysis.trust_score,
            detected_lang=detected_lang_label,
            top_words=pred.words if pred.words else []
        )
        st.download_button(
            label="📥 Download Full Authenticity Assessment Report (.txt)",
            data=report_text,
            file_name=f"truthguard_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================================================= PAGE 3: 🔎 SOURCE VERIFICATION
elif selected_page == "🔎 Source Verification":
    st.markdown("""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>FACT-CHECK REGISTRY · DOMAIN REPUTATION</span>
      <h1>Publisher & Domain <span class="grad-text">Reputation Index</span></h1>
      <p>Analyze domain credibility and verify publisher authenticity against accredited global and Indian fact-checking registries.</p>
    </div>""", unsafe_allow_html=True)

    v_col1, v_col2 = st.columns([3, 1])
    with v_col1:
        test_domain_input = st.text_input("Enter URL or Domain", placeholder="e.g. pib.gov.in, reuters.com, fake-news.co", label_visibility="collapsed")
    with v_col2:
        check_domain_btn = st.button("🔎 Test Publisher Domain", use_container_width=True)

    if test_domain_input or check_domain_btn:
        clean_dom = extract_domain(test_domain_input)
        res = analyze_source(clean_dom, "REAL", 0.90)
        col = "#059669" if res.credibility in ("High", "Medium-High") else ("#d97706" if res.credibility == "Satire" else "#dc2626")
        bg_col = "#ecfdf5" if res.credibility in ("High", "Medium-High") else ("#fffbeb" if res.credibility == "Satire" else "#fef2f2")

        st.markdown(f"""
        <div class="card-light" style="border-color:{col}; background:{bg_col}; margin-top:1rem;">
            <div style="font-size:0.8rem; font-weight:700; color:{col}; text-transform:uppercase;">DOMAIN CREDIBILITY ASSESSMENT</div>
            <div style="font-family:'Outfit'; font-size:1.8rem; font-weight:800; color:{col}; margin:0.3rem 0;">{res.source_name} ({res.domain})</div>
            <div style="font-size:1.05rem; color:#1e293b; margin-bottom:0.6rem;"><b>Rating:</b> {res.credibility} · <b>Trust Score:</b> {res.trust_score * 100:.0f}%</div>
            <p style="color:#334155; margin:0; line-height:1.6;">{res.description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Accredited Fact-Check Registry Directory")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card-light'><b>🟢 Verified News & Government Agencies</b>", unsafe_allow_html=True)
        for dom, (name, cred, desc) in list(TRUSTED_DOMAINS.items())[:6]:
            st.markdown(f"• **{name}** (`{dom}`): {desc}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card-light'><b>🔴 Known Clickbait & Satirical Outlets</b>", unsafe_allow_html=True)
        for dom, (name, cred, desc) in list(UNTRUSTED_DOMAINS.items()):
            st.markdown(f"• **{name}** (`{dom}`): {desc}")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================= PAGE 4: 📊 MODEL COMPARISON
elif selected_page == "📊 Model Comparison":
    st.markdown("""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>EMPIRICAL BENCHMARKS · HELD-OUT TEST SET</span>
      <h1>Model Performance <span class="grad-text">& Leaderboard</span></h1>
      <p>Quantitative evaluation comparing BERT Transformer, BiLSTM, Simple RNN, and TF-IDF baselines on test articles.</p>
    </div>""", unsafe_allow_html=True)

    if not reports:
        st.info("No evaluation reports found in the reports directory.")
    else:
        order = [k for k in ["tfidf_logreg", "bert", "lstm", "rnn"] if k in reports]
        sel = model_name if model_name in reports else (order[0] if order else list(reports.keys())[0])
        m = reports[sel]

        st.markdown(f"<h3 style='font-family:Outfit; font-size:1.45rem; font-weight:800; color:#0f172a'>Test Set Evaluation — {PRETTY.get(sel, sel)}</h3>"
                    f"<div style='color:#64748b; margin-bottom:1rem;'>Evaluated on {m['n_test']:,} test articles · trained on {m.get('train_size', 0):,} samples</div>",
                    unsafe_allow_html=True)

        st.markdown("<div style='display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1rem 0;'>" + "".join(
            f"<div class='card-light'><div style='font-size:0.8rem; color:#64748b; font-weight:700; text-transform:uppercase;'>{k}</div><div style='font-family:Outfit; font-size:1.85rem; font-weight:800; color:#0f172a; margin-top:0.2rem;'>{m[key] * 100:.2f}%</div></div>"
            for k, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 Score", "f1")]) + "</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1.25], gap="large")
        c1.plotly_chart(cm_fig(m["confusion_matrix"], f"Confusion Matrix — {PRETTY.get(sel, sel)}"),
                        use_container_width=True, config={"displayModeBar": False})

        mets = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
        palette = ["#4f46e5", "#059669", "#10b981", "#0284c7"]
        fig = go.Figure()
        for i, k in enumerate(order):
            fig.add_bar(name=PRETTY.get(k, k), x=[a for a, _ in mets], y=[reports[k][b] for _, b in mets],
                        marker_color=palette[i % len(palette)], text=[f"{reports[k][b]:.3f}" for _, b in mets],
                        textposition="outside", textfont=dict(size=10, color="#0f172a"))
        lo = min(reports[k][b] for k in order for _, b in mets) if order else 0.5
        fig.update_yaxes(range=[max(0, lo - 0.08), 1.01])
        fig.update_layout(barmode="group", title=dict(text="Comparative Model Metrics", font=dict(family="Outfit", size=15, color="#0f172a")))
        c2.plotly_chart(plot_layout(fig, 340), use_container_width=True, config={"displayModeBar": False})

        rows = []
        for k in order:
            r = reports[k]
            rows.append({"Model Architecture": PRETTY.get(k, k), "Accuracy": f"{r['accuracy'] * 100:.2f}%",
                         "Precision (Fake)": f"{r['precision'] * 100:.2f}%", "Recall (Fake)": f"{r['recall'] * 100:.2f}%",
                         "F1 Score": f"{r['f1'] * 100:.2f}%", "ROC-AUC": f"{r.get('roc_auc', 0):.4f}",
                         "Test Samples": f"{r['n_test']:,}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================================= PAGE 5: 🧠 EXPLAINABLE AI
elif selected_page == "🧠 Explainable AI":
    st.markdown("""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>INTEGRATED GRADIENTS · ATTRIBUTION METRICS</span>
      <h1>Explainable AI <span class="grad-text">(XAI) Explorer</span></h1>
      <p>Understand neural network decisions through gradient-based attribution that traces predictions back to individual words.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-light">
        <h3>How Integrated Gradients Works in TruthGuard</h3>
        <p style="color:#334155; line-height:1.7;">
        <b>Integrated Gradients (IG)</b> satisfies fundamental interpretability axioms (<i>Completeness</i> and <i>Implementation Invariance</i>).
        By computing the path integral of gradients along the straight line from a baseline embedding (empty/padding embedding) to the input token embeddings,
        the model calculates exactly how much each word shifts the classification score towards <b>FAKE (positive attribution)</b> or <b>REAL (negative attribution)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    xai_input = st.text_area("Test Sentence for XAI Attribution",
                             value="BREAKING: Shocking secret government report claims water can cure every disease without medicine!",
                             height=100)
    if st.button("🧠 Compute Token Attributions", type="primary", use_container_width=True):
        pred_x = get_predictor(str(models[model_name])).predict(xai_input, explain=True, ig_steps=32)
        if pred_x.words:
            c1, c2 = st.columns([1.6, 1], gap="large")
            c1.markdown(highlight_html(pred_x.words, L), unsafe_allow_html=True)
            c2.plotly_chart(top_words_fig(pred_x.words, "Influential Tokens Breakdown"), use_container_width=True, config={"displayModeBar": False})
            st.info("ℹ️ **Explanation Summary:** These highlighted tokens exerted the strongest mathematical gradient pull on the neural classifier's decision.")


# ============================================================================= PAGE 6: 📜 PREDICTION HISTORY
elif selected_page == "📜 Prediction History":
    st.markdown("""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>SESSION AUDIT LOG · RECENT ANALYSES</span>
      <h1>Session Prediction <span class="grad-text">History</span></h1>
      <p>Review real-time verification logs performed during your current active session.</p>
    </div>""", unsafe_allow_html=True)

    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No predictions recorded in this session yet. Navigate to 🔍 News Detector to run your first verification.")
    else:
        h_df = pd.DataFrame(hist)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Analyzed", len(hist))
        c2.metric("Real Predictions", len([h for h in hist if h['label'] == 'REAL']))
        c3.metric("Fake / Uncertain", len([h for h in hist if h['label'] != 'REAL']))

        st.dataframe(h_df[["time", "text", "label", "confidence", "risk", "model", "source"]], use_container_width=True, hide_index=True)

        b_c1, b_c2 = st.columns([1, 1])
        with b_c1:
            csv_data = h_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export History to CSV", data=csv_data, file_name="truthguard_prediction_history.csv", mime="text/csv", use_container_width=True)
        with b_c2:
            if st.button("🧹 Clear Session History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()


# ============================================================================= PAGE 7: 🧩 HOW IT WORKS
elif selected_page == "🧩 How It Works":
    st.markdown("""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>PIPELINE ARCHITECTURE · END-TO-END WORKFLOW</span>
      <h1>TruthGuard NLP <span class="grad-text">Pipeline Architecture</span></h1>
      <p>From raw multilingual text input to neural encoding, attention mechanisms, and explainable decision attributions.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-light">
        <h3>End-to-End Verification Flow</h3>
        <div style="line-height:2; color:#334155; font-size:1rem;">
        <b>1. Multilingual Ingestion</b> — Ingests news headlines, full articles, PDFs, or live URLs in English, Hindi, or Marathi.<br>
        <b>2. Text Cleaning & Normalization</b> — Cleans datelines (e.g. "BERLIN - Reuters"), social media handles, and normalizes unicode scripts.<br>
        <b>3. Tokenization Subsystems</b> — Subword WordPiece tokenization for BERT, Byte-Pair Encoding (BPE) for RoBERTa, and custom frequency dictionaries for RNN/BiLSTM.<br>
        <b>4. Neural Context Encoding</b> — Multi-head self-attention layers compute bidirectional contextual representations across full sentences.<br>
        <b>5. Softmax Classification Head</b> — Produces calibrated probability distributions for <b>FAKE</b> and <b>REAL</b> classes.<br>
        <b>6. Integrated Gradients Attribution</b> — Traces gradients from output back to input embeddings to explain model logic token by token.<br>
        <b>7. Source Registry Cross-Check</b> — Evaluates publisher domain reputation against accredited fact-checking databases.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================= PAGE 8: ℹ️ ABOUT PROJECT
elif selected_page == "ℹ️ About Project":
    st.markdown("""
    <div class="hero-banner">
      <span class="badge-pill"><span class="dot"></span>ACADEMIC OVERVIEW · RESEARCH METADATA</span>
      <h1>About TruthGuard <span class="grad-text">Project</span></h1>
      <p>A comprehensive research and demonstration project for explainable multilingual NLP fake news detection.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-light">
        <h3>Project Specifications</h3>
        <table style="width:100%; border-collapse:collapse; font-size:0.95rem; color:#1e293b;">
            <tr style="border-bottom:1px solid #e2e8f0; height:36px;"><td style="width:220px; font-weight:700;">Project Title</td><td>🛡️ TruthGuard: Multilingual & Explainable Fake News Detection System</td></tr>
            <tr style="border-bottom:1px solid #e2e8f0; height:36px;"><td style="font-weight:700;">Objective</td><td>Automate detection of misinformation, clickbait, and manipulated news across English and Indian regional languages with mathematical explainability.</td></tr>
            <tr style="border-bottom:1px solid #e2e8f0; height:36px;"><td style="font-weight:700;">Dataset</td><td>ISOT Fake & Real News Benchmark (44,898 articles) + Curated Multilingual Hindi/Marathi Fact Corpus.</td></tr>
            <tr style="border-bottom:1px solid #e2e8f0; height:36px;"><td style="font-weight:700;">NLP Technologies</td><td>PyTorch, Hugging Face Transformers, WordPiece/BPE Tokenizers, Integrated Gradients XAI, TF-IDF n-grams.</td></tr>
            <tr style="border-bottom:1px solid #e2e8f0; height:36px;"><td style="font-weight:700;">Web Framework</td><td>Streamlit, Plotly Interactive Visualizations, PyPDF, BeautifulSoup4.</td></tr>
            <tr style="border-bottom:1px solid #e2e8f0; height:36px;"><td style="font-weight:700;">College / Department</td><td>Department of Computer Science & Engineering (Final Year Major Project)</td></tr>
            <tr style="height:36px;"><td style="font-weight:700;">Academic Mentor</td><td>Project Advisor / Review Committee Placeholder</td></tr>
        </table>
    </div>

    <div class="card-light" style="border-left:4px solid #4f46e5;">
        <h3 style="color:#4f46e5;">⚠️ Academic & Ethical Disclaimer</h3>
        <p style="color:#475569; line-height:1.6; margin:0;">
        This system provides an AI-based probabilistic assessment and does not establish the absolute legal or factual truth of a news article.
        Predictions are based on statistical linguistic patterns learned from labeled training corpora. Always verify critical claims using accredited fact-checking agencies.
        </p>
    </div>
    """, unsafe_allow_html=True)

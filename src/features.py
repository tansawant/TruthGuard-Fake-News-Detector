"""Helper utilities for TruthGuard: PDF extraction, URL scraping, Language detection, Claim splitting, and Report Generation."""
from __future__ import annotations

import io
import re
from typing import List, Tuple
from urllib.parse import urlparse
import datetime

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


def detect_language(text: str) -> Tuple[str, str]:
    """Detects whether text is English, Hindi, or Marathi based on script and marker tokens.
    
    Returns (lang_code, lang_name_with_flag).
    """
    if not text or not text.strip():
        return "en", "English 🇬🇧"
    
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    total_chars = len(re.findall(r"\w", text)) or 1
    
    if devanagari_chars / total_chars > 0.25:
        # Check Marathi distinct words vs Hindi
        marathi_markers = {"आहे", "नाही", "झाले", "केले", "म्हणाले", "शेतकरी", "यांनी", "आणि", "झाली", "झाला"}
        hindi_markers = {"है", "नहीं", "कहा", "किया", "होगा", "योजना", "भारत", "सरकार", "और", "हुए"}
        
        words = set(text.split())
        mr_score = len(words.intersection(marathi_markers))
        hi_score = len(words.intersection(hindi_markers))
        
        if mr_score > hi_score:
            return "mr", "Marathi 🇮🇳"
        return "hi", "Hindi 🇮🇳"
        
    return "en", "English 🇬🇧"


def extract_claims(text: str, max_claims: int = 6) -> List[str]:
    """Extract distinct sentences/claims from an article for granular claim-by-claim verification."""
    if not text:
        return []
    # Split by standard punctuation and devanagari purna viram (।)
    raw_sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    claims = []
    for s in raw_sentences:
        clean_s = s.strip()
        if len(clean_s.split()) >= 4 and len(clean_s) >= 25:
            claims.append(clean_s)
    return claims[:max_claims]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from uploaded PDF file bytes using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages_text.append(t)
        return "\n\n".join(pages_text).strip()
    except Exception as e:
        return f"Error reading PDF: {e}"


def extract_text_from_url(url: str, timeout_sec: int = 6) -> Tuple[str, str, str, str]:
    """Scrapes headline and content from a given news article URL.
    
    Returns (title, body_text, domain, error_msg).
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
        
    domain = urlparse(url).netloc.lower().replace("www.", "")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Remove script, style, and navigation tags
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
            
        title = ""
        if soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)
            
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        body_text = "\n\n".join(paragraphs[:12])
        
        full_text = f"{title}\n\n{body_text}".strip() if title else body_text
        return title, full_text, domain, ""
    except Exception as e:
        return "", "", domain, f"Failed to retrieve URL: {str(e)}"


def assess_risk(label: str, confidence: float) -> Tuple[str, str, str, str]:
    """Calculates risk level and uncertainty for academic & demo completeness.
    
    Returns (status_label, risk_level, risk_color, risk_description).
    """
    if confidence < 0.65:
        return (
            "UNCERTAIN",
            "MEDIUM",
            "#d97706",
            "⚠️ Low confidence assessment (50–64%). Mixed contextual signals detected. Manual fact-checking strongly recommended."
        )
    if label == "REAL":
        return (
            "REAL",
            "LOW",
            "#059669",
            "🟢 High authenticity probability. Low misinformation risk detected based on linguistic features."
        )
    return (
        "FAKE",
        "HIGH" if confidence >= 0.85 else "MEDIUM-HIGH",
        "#dc2626",
        "🔴 High misinformation probability. Strong deceptive patterns, hyperbole, or clickbait signals detected."
    )


def generate_assessment_report(
    news_text: str,
    pred_label: str,
    confidence: float,
    model_name: str,
    risk_level: str,
    risk_desc: str,
    source_domain: str,
    source_trust_score: float,
    detected_lang: str,
    top_words: list
) -> str:
    """Builds a formatted text/markdown assessment report for download."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    words_summary = ", ".join([f"{w} ({s:+.2f})" for w, s in top_words[:8]]) if top_words else "N/A"
    
    report = f"""================================================================================
🛡️ TRUTHGUARD AI — FAKE NEWS AUTHENTICITY ASSESSMENT REPORT
Generated: {now_str}
================================================================================

1. EXECUTIVE ASSESSMENT
--------------------------------------------------------------------------------
• Primary Verdict:        {pred_label}
• Model Confidence Score: {confidence * 100:.1f}%
• Misinformation Risk:    {risk_level}
• Active AI Architecture: {model_name}
• Detected Language:      {detected_lang}
• Risk Advisory:          {risk_desc}

2. SOURCE & FACT-CHECK REPUTATION
--------------------------------------------------------------------------------
• Publisher Domain:       {source_domain if source_domain else 'Not provided'}
• Domain Trust Score:     {source_trust_score * 100:.1f}% (Based on Fact-Check Index)

3. EXPLAINABLE AI (XAI) TOP INFLUENTIAL TOKENS
--------------------------------------------------------------------------------
• Significant Tokens:     {words_summary}
(Positive values push toward FAKE, Negative values push toward REAL)

4. INPUT NEWS ARTICLE
--------------------------------------------------------------------------------
{news_text}

================================================================================
⚠️ ACADEMIC & RESEARCH DISCLAIMER:
This system provides an AI-based probabilistic assessment and does not establish
the absolute legal or factual truth of a news article. Verify critical claims
using certified fact-checking agencies (e.g. PIB Fact Check, Reuters, BBC, Snopes).
================================================================================
"""
    return report

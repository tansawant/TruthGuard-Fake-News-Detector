"""Source verification, domain credibility assessment, and fact-checking validation module."""
from __future__ import annotations

import re
from urllib.parse import urlparse
from dataclasses import dataclass

TRUSTED_DOMAINS = {
    "reuters.com": ("Reuters", "High", "Global news agency known for strict factual reporting."),
    "bbc.com": ("BBC News", "High", "Public service broadcaster with high editorial standards."),
    "pib.gov.in": ("Press Information Bureau (Govt of India)", "High", "Official Indian Government nodal agency."),
    "altnews.in": ("Alt News", "High", "Certified Indian non-profit fact-checking portal."),
    "boomlive.in": ("BOOM FactCheck", "High", "IFCN-certified fact-checking organization."),
    "snopes.com": ("Snopes", "High", "Established independent fact-checking site."),
    "factcheck.org": ("FactCheck.org", "High", "Nonpartisan fact-checking website."),
    "nasa.gov": ("NASA Official", "High", "Official space research organization."),
    "ndtv.com": ("NDTV", "Medium-High", "Major Indian news network."),
    "thehindu.com": ("The Hindu", "High", "Reputable Indian national newspaper."),
    "timesofindia.indiatimes.com": ("Times of India", "Medium-High", "Major Indian news outlet."),
    "indianexpress.com": ("Indian Express", "High", "Reputable Indian daily newspaper."),
    "amarujala.com": ("Amar Ujala", "Medium-High", "Major Hindi daily newspaper."),
    "dainikbhaskar.com": ("Dainik Bhaskar", "Medium-High", "Leading Hindi newspaper."),
}

UNTRUSTED_DOMAINS = {
    "fake-news.co": ("FakeNews Portal", "Low", "Known clickbait site with unverified claims."),
    "satirewire.com": ("Satire Wire", "Satire", "Explicit satire and fictional stories."),
    "weeklyworldnews.com": ("Weekly World News", "Satire/Fake", "Tabloid publication featuring fictional stories."),
    "theonion.com": ("The Onion", "Satire", "Satirical news organization."),
    "worldnewsdailyreport.com": ("World News Daily Report", "Satire/Fake", "Fictional news site."),
}


@dataclass
class SourceAnalysis:
    domain: str
    source_name: str
    credibility: str  # "High" | "Medium" | "Low" | "Unknown" | "Satire"
    trust_score: float  # 0.0 to 1.0
    description: str
    is_validated_source: bool
    combined_confidence: float  # Adjusted confidence considering model + source


def extract_domain(input_str: str) -> str:
    """Extract domain from a URL or return cleaned string."""
    input_str = input_str.strip()
    if input_str.startswith("http://") or input_str.startswith("https://"):
        parsed = urlparse(input_str)
        return parsed.netloc.lower().replace("www.", "")
    match = re.search(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", input_str)
    if match:
        return match.group(1).lower()
    return input_str.lower()


def analyze_source(source_input: str, model_pred: str, model_confidence: float) -> SourceAnalysis:
    """Analyze news source domain and combine with model confidence."""
    if not source_input or len(source_input.strip()) == 0:
        return SourceAnalysis(
            domain="Not Provided",
            source_name="Unknown Source",
            credibility="Unknown",
            trust_score=0.5,
            description="No source URL or publisher provided for domain verification.",
            is_validated_source=False,
            combined_confidence=model_confidence
        )

    domain = extract_domain(source_input)

    # Check in trusted domains
    for d, (name, cred, desc) in TRUSTED_DOMAINS.items():
        if d in domain or domain in d:
            trust_score = 0.95 if cred == "High" else 0.85
            # If model says REAL and source is high trust -> boost confidence
            # If model says FAKE but source is high trust -> flag discrepancy
            adj_conf = (model_confidence * 0.7) + (trust_score * 0.3) if model_pred == "REAL" else max(0.1, model_confidence * 0.7)
            return SourceAnalysis(
                domain=domain,
                source_name=name,
                credibility=cred,
                trust_score=trust_score,
                description=desc,
                is_validated_source=True,
                combined_confidence=round(adj_conf, 4)
            )

    # Check in untrusted / satirical domains
    for d, (name, cred, desc) in UNTRUSTED_DOMAINS.items():
        if d in domain or domain in d:
            trust_score = 0.1
            adj_conf = (model_confidence * 0.7) + (0.9 * 0.3) if model_pred == "FAKE" else max(0.1, model_confidence * 0.5)
            return SourceAnalysis(
                domain=domain,
                source_name=name,
                credibility=cred,
                trust_score=trust_score,
                description=desc,
                is_validated_source=True,
                combined_confidence=round(adj_conf, 4)
            )

    # General domain fallback
    return SourceAnalysis(
        domain=domain,
        source_name=domain.capitalize(),
        credibility="Unverified",
        trust_score=0.5,
        description="Domain is active but not listed in known verified news index.",
        is_validated_source=False,
        combined_confidence=model_confidence
    )

import random

def score_url(url: str) -> float:
    # TODO: Replace with real ML model (TF-IDF, recency, backlinks, etc.)
    # For now, random score for demo
    return round(random.uniform(0.5, 1.0), 3)

def score_pdf(content: bytes) -> float:
    # TODO: Replace with real ML model (length, novelty, etc.)
    return round(random.uniform(0.5, 1.0), 3) 
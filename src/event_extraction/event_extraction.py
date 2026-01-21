from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _load_spacy(model: str = "en_core_web_sm"):
    """Load spaCy model with a friendlier error message."""
    try:
        return spacy.load(model)
    except OSError as e:
        raise SystemExit(
            f"spaCy model '{model}' not found. Install it with:\n\n"
            f"  python -m spacy download {model}\n"
        ) from e


def extract_events_semantic(
    texts: List[str],
    event_phrases: Optional[List[str]] = None,
    *,
    spacy_model: str = "en_core_web_sm",
) -> List[Dict]:
    """Detect event-like phrases using TF-IDF similarity + extract GPE entities via spaCy.

    Args:
        texts: Input strings (headlines, short texts).
        event_phrases: Canonical event phrases to match against (defaults provided).
        spacy_model: spaCy model name for NER.

    Returns:
        List of dicts with headline, predicted event, and extracted entities.
    """
    if not texts:
        return []

    # default canonical event templates
    if event_phrases is None:
        event_phrases = [
            "explosion",
            "earthquake",
            "protest",
            "flood",
            "wildfire",
            "storm",
            "attack",
            "shooting",
            "strike",
            "evacuation",
        ]

    nlp = _load_spacy(spacy_model)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts + event_phrases)

    text_vectors = tfidf[: len(texts)]
    event_vectors = tfidf[len(texts) :]

    sims = cosine_similarity(text_vectors, event_vectors)
    best_idx = np.argmax(sims, axis=1)

    results: List[Dict] = []
    for text, idx in zip(texts, best_idx):
        doc = nlp(text)
        gpes = sorted({ent.text for ent in doc.ents if ent.label_ == "GPE"})
        results.append(
            {
                "headline": text,
                "event": event_phrases[int(idx)],
                "entities": gpes,
            }
        )
    return results


def extract_events_string_match(
    texts: List[str],
    event_dict: Dict[str, str],
    *,
    spacy_model: str = "en_core_web_sm",
) -> List[Dict]:
    """Detect events by keyword regex matching + extract GPE entities via spaCy."""
    if not texts:
        return []

    nlp = _load_spacy(spacy_model)

    results: List[Dict] = []
    for text in texts:
        events: List[str] = []
        for word, label in event_dict.items():
            if re.search(rf"\b{re.escape(word)}\b", text.lower()):
                events.append(label)

        doc = nlp(text)
        gpes = sorted({ent.text for ent in doc.ents if ent.label_ == "GPE"})
        results.append(
            {
                "headline": text,
                "events": sorted(set(events)),
                "entities": gpes,
            }
        )
    return results


def _demo():
    headlines = [
        "Massive explosion rocks Beirut port area",
        "Thousands protest government corruption in the capital",
        "Earthquake of magnitude 6.7 hits Japan coast",
        "Wildfire forces evacuation of coastal town",
    ]

    event_keywords = {
        "explosion": "Explosion",
        "earthquake": "Earthquake",
        "protest": "Protest",
        "wildfire": "Wildfire",
        "evacuation": "Evacuation",
    }

    print("Semantic extraction:\n", pd.DataFrame(extract_events_semantic(headlines)))
    print("\nKeyword extraction:\n", pd.DataFrame(extract_events_string_match(headlines, event_keywords)))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Event extraction demos (semantic + keyword match).")

    p.add_argument("--demo", action="store_true", help="Run built-in demo headlines.")
    p.add_argument("--text", action="append", default=[], help="Add an input text/headline (repeatable).")

    p.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name (default: en_core_web_sm).")

    args = p.parse_args(argv)

    if args.demo:
        _demo()
        return 0

    if not args.text:
        p.error("Provide --demo or at least one --text.")

    df_sem = pd.DataFrame(extract_events_semantic(args.text, spacy_model=args.spacy_model))
    print(df_sem.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
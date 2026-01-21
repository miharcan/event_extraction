from __future__ import annotations

import argparse
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


SAMPLE_DATA = [
    {"id": 1, "text": "Armed clashes reported between groups in Khartoum, Sudan."},
    {"id": 2, "text": "Floods displaced thousands in Mogadishu, Somalia."},
    {"id": 3, "text": "A cyclone caused widespread damage along the coast of Mozambique."},
    {"id": 4, "text": "Earthquake tremors felt in northern Turkey; buildings evacuated."},
    {"id": 5, "text": "Protesters gathered in the capital demanding political reforms."},
]


def _rank_by_embeddings(texts: List[str], query: str, model_name: str) -> np.ndarray:
    """Return cosine similarity scores using SentenceTransformers."""
    from sentence_transformers import SentenceTransformer  # heavy import

    model = SentenceTransformer(model_name)
    query_emb = model.encode([query], normalize_embeddings=True)
    text_emb = model.encode(texts, normalize_embeddings=True)
    return cosine_similarity(query_emb, text_emb)[0]


def _explain_relevance(query: str, text: str, hf_model: str, max_new_tokens: int = 120) -> str:
    """Generate a short explanation via a HuggingFace text-generation pipeline."""
    from transformers import pipeline  # heavy import

    explainer = pipeline("text-generation", model=hf_model)
    prompt = f"Explain why the following text is relevant to '{query}':\n\n{text}\n\nExplanation:"
    out = explainer(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)[0]["generated_text"]
    out = re.sub(r"\s+", " ", out).strip()
    m = re.search(r"Explanation:(.*)", out)
    return (m.group(1).strip() if m else out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Semantic ranking of short event texts (with optional explanations).")

    p.add_argument("--query", required=True, help="Search query, e.g. 'flood displacement'.")
    p.add_argument("--top-k", type=int, default=5, help="How many results to show.")
    p.add_argument("--no-explain", action="store_true", help="Disable explanation generation.")
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name.",
    )
    p.add_argument(
        "--explain-model",
        default="EleutherAI/gpt-neo-125M",
        help="HuggingFace text-generation model name used for explanations.",
    )

    args = p.parse_args(argv)

    df = pd.DataFrame(SAMPLE_DATA)
    texts = df["text"].tolist()

    scores = _rank_by_embeddings(texts, args.query, args.embedding_model)
    df["score"] = scores
    df = df.sort_values("score", ascending=False).head(args.top_k).reset_index(drop=True)

    if not args.no_explain:
        df["explanation"] = [
            _explain_relevance(args.query, t, args.explain_model) for t in df["text"].tolist()
        ]

    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
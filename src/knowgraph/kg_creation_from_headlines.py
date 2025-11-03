import os
import re
import requests
import networkx as nx
import spacy
import matplotlib.pyplot as plt


# simple .env loading: prefer ../../.env, otherwise try default locations
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(path=None):
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return True
        except Exception:
            return False

_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    # try default loader (python-dotenv searches common places) or no-op fallback
    load_dotenv()

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise RuntimeError("NEWSAPI_KEY not found. Create a .env with NEWSAPI_KEY=your_key or set the environment variable.")

# ---- simple configuration ----
QUERY = "Trump"
SOURCES = "bbc-news,abc-news,al-jazeera-english,associated-press"
OUTPUT_DIR = "/home/miharc/work/code/event_extraction/src/knowgraph/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUTPUT_DIR, "kg_headlines.png")

# very small event keyword map (easy to extend)
EVENT_KEYWORDS = {
    "explosion": "Disaster",
    "blast": "Disaster",
    "protest": "Civil Unrest",
    "demonstration": "Civil Unrest",
    "earthquake": "Natural Disaster",
}

# load spaCy once
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # fallback to a blank English model if the small model isn't installed;
    # functionality will be reduced (no NER) but code stays simple.
    nlp = spacy.blank("en")

# ---- fetching headlines (simple, clear) ----
def fetch_headlines(query, sources, api_key, language="en", page_size=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sources": sources,
        "language": language,
        "pageSize": page_size,
        "sortBy": "relevancy",
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("Failed to fetch headlines:", e)
        return []
    articles = data.get("articles", [])
    texts = []
    for a in articles:
        desc = a.get("description") or ""
        cont = a.get("content") or ""
        text = f"{desc} {cont}".strip()
        # strip common bracketed noise like "[Read more]"
        text = re.sub(r"\[.*?\]", "", text).strip()
        if text:
            texts.append(text)
    return texts

# ---- simple event extractor using keywords + NER entities ----
def extract_events_and_entities(texts, keywords):
    results = []
    kw_items = [(k.lower(), v) for k, v in keywords.items()]
    for t in texts:
        low = t.lower()
        found = sorted({label for k, label in kw_items if re.search(rf"\b{re.escape(k)}\b", low)})
        doc = nlp(t)
        ents = []
        # If the model has NER, collect GPE/PERSON/ORG; otherwise collect tokens heuristically
        if doc.ents:
            ents = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "PERSON", "ORG")]
        else:
            # very simple fallback: proper nouns from the sentence
            ents = [tok.text for tok in doc if tok.pos_ == "PROPN"]
        results.append({"headline": t, "events": found, "entities": sorted(set(ents))})
    return results

# small helper to build a short phrase for a token (compounds + token)
def phrase_for_token(tok):
    parts = []
    for child in tok.lefts:
        if child.dep_ in ("compound", "amod", "det"):
            parts.append(child.text)
    parts.append(tok.text)
    for child in tok.rights:
        if child.dep_ in ("compound",):
            parts.append(child.text)
    return " ".join(parts).strip()

# ---- build a very simple KG from sentences: subject -> object edges labeled by verb ----
def build_kg_from_texts(texts):
    kg = nx.DiGraph()
    if not texts:
        return kg
    # process each text separately to avoid overly connecting unrelated sentences
    for t in texts:
        doc = nlp(t)
        # add entities as nodes for clarity (if available)
        for ent in doc.ents:
            kg.add_node(ent.text, type=ent.label_)
        for sent in doc.sents:
            for tok in sent:
                if tok.pos_ == "VERB":
                    subj = None
                    obj = None
                    for ch in tok.children:
                        if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN", "PRON"):
                            subj = phrase_for_token(ch)
                        if ch.dep_ in ("dobj", "obj", "pobj") and ch.pos_ in ("NOUN", "PROPN", "PRON"):
                            obj = phrase_for_token(ch)
                    # also allow prepositional object as object if no direct object
                    if not obj:
                        for ch in tok.children:
                            if ch.dep_ == "prep":
                                for pc in ch.children:
                                    if pc.dep_ == "pobj" and pc.pos_ in ("NOUN", "PROPN"):
                                        obj = phrase_for_token(pc)
                                        break
                    if subj:
                        kg.add_node(subj, type="ARG")
                    if obj:
                        kg.add_node(obj, type="ARG")
                    if subj and obj:
                        rel = tok.lemma_.lower()
                        kg.add_edge(subj, obj, relation=rel, sentence=sent.text.strip())
    # Optionally keep only the largest connected component to reduce noise
    if kg.number_of_nodes() > 0:
        comps = list(nx.weakly_connected_components(kg))
        if len(comps) > 1:
            largest = max(comps, key=len)
            rm = set(kg.nodes()) - set(largest)
            kg.remove_nodes_from(rm)
    return kg

# ---- draw and save the KG with a simple layout ----
def draw_and_save_kg(kg, out_path):
    if kg.number_of_nodes() == 0:
        print("No nodes to draw.")
        return
    plt.figure(figsize=(12, 8))
    try:
        pos = nx.spring_layout(kg, seed=42)
    except Exception:
        pos = nx.random_layout(kg, seed=42)
    # node colors by type attribute
    types = nx.get_node_attributes(kg, "type")
    unique = sorted(set(types.values()))
    color_map = {}
    palette = plt.cm.get_cmap("tab10")
    for i, t in enumerate(unique):
        color_map[t] = palette(i % 10)
    node_colors = [color_map.get(types.get(n), (0.8, 0.8, 0.8)) for n in kg.nodes()]
    nx.draw_networkx_nodes(kg, pos, node_color=node_colors, node_size=700, alpha=0.9)
    nx.draw_networkx_labels(kg, pos, font_size=9)
    nx.draw_networkx_edges(kg, pos, arrowstyle="->", arrowsize=10, edge_color="#444444")
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in kg.edges(data=True)}
    if edge_labels:
        nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color="gray", font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved KG image: {out_path}")

# ---- main flow ----
def main():
    texts = fetch_headlines(QUERY, SOURCES, NEWSAPI_KEY)
    if not texts:
        print("No articles fetched.")
        return
    events = extract_events_and_entities(texts, EVENT_KEYWORDS)
    for e in events:
        if e["events"]:
            print("Detected events:", e["events"], "Entities:", e["entities"])
    kg = build_kg_from_texts(texts)
    print(f"Built KG with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges.")
    # small sample printouts
    for n, d in list(kg.nodes(data=True))[:20]:
        print(n, d)
    for u, v, d in list(kg.edges(data=True))[:20]:
        print(f"{u} -[{d.get('relation')}]-> {v}")
    draw_and_save_kg(kg, OUT_PATH)

if __name__ == "__main__":
    main()

import os
import re
import requests
import networkx as nx
import spacy
import matplotlib.pyplot as plt
from dotenv import load_dotenv 
from newsapi import NewsApiClient
from networkx.algorithms import community as nx_comm
import os
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
NEWSAPI_KEY = (os.getenv("NEWSAPI_KEY"))
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
r = newsapi.get_sources()

considered_langs = ['en','de', 'it', 'es', 'fr']

nlp = spacy.load("en_core_web_sm")

QUERY = "Donald Trump"

def fetch_headlines(query, sources, api_key, from_date, to_date, language, page_size=20):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sources": sources,
        "language": language,
        "pageSize": page_size,
        "from": from_date,
        "to": to_date,
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
        # text = f"{desc}. {cont}"
        text = cont
        text = re.sub(r"\[.*?\]", "", text).strip()
        texts.append(text)
    return texts

def kg_creation(texts):
    kg = nx.DiGraph()

    kg_creation.min_edges = 1  # optional: nodes must have degree >= min_edges
    min_edges = getattr(kg_creation, "min_edges", None)

    # extract simple SVO triples and add them to the kg
    subj_deps = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
    obj_deps = {"dobj", "dative", "attr", "oprd", "pobj", "obj"}
    triples = []

    for t in texts:
        sentences = [i for i in nlp(t).sents]
        for s in sentences:
            doc = s.as_doc()

            def phrase_text(tok):
                # prefer the noun-chunk root if available
                root = tok
                for nc in doc.noun_chunks:
                    if tok.i >= nc.start and tok.i < nc.end:
                        root = nc.root
                        break

                # For proper nouns, keep contiguous PROPNs (e.g. "New York Times")
                if root.pos_ == "PROPN":
                    start = root.i
                    end = root.i
                    while start - 1 >= 0 and doc[start - 1].pos_ == "PROPN":
                        start -= 1
                    while end + 1 < len(doc) and doc[end + 1].pos_ == "PROPN":
                        end += 1
                    return doc[start : end + 1].text

                # For common nouns, include only left-side compounds/adjectival modifiers + head
                if root.pos_ in {"NOUN", "PROPN"}:
                    left_mods = [t for t in root.lefts if t.dep_ in {"compound", "amod"}]
                    left_mods = sorted(left_mods, key=lambda t: t.i)
                    parts = [t.text for t in left_mods] + [root.text]
                    return " ".join(parts)

                # fallback to token text
                return root.text

            def normalize_arg(tok):
                # exclude pronouns and auxiliaries
                if tok.pos_ in {"PRON", "AUX"}:
                    return None
                # prefer noun-chunk root if present (gives canonical head)
                for nc in doc.noun_chunks:
                    if tok.i >= nc.start and tok.i < nc.end:
                        root = nc.root
                        if root.pos_ in {"PRON", "AUX"}:
                            return None
                        if root.ent_type_ or root.pos_ in {"NOUN", "PROPN"}:
                            return root
                        return None
                # keep named entities or nouns/proper nouns
                if tok.ent_type_ or tok.pos_ in {"NOUN", "PROPN"}:
                    return tok
                return None

            def filter_and_dedupe(candidates):
                seen = set()
                out = []
                for c in candidates:
                    n = normalize_arg(c)
                    if n and n.i not in seen:
                        seen.add(n.i)
                        out.append(n)
                return out

            for token in doc:
                # skip auxiliaries entirely (they should not be treated as main relations)
                if token.pos_ == "AUX":
                    continue
                # consider main verbs (and root tokens) but not auxiliaries
                if not (token.pos_ == "VERB" or token.dep_ == "ROOT"):
                    continue

                raw_subjects = [c for c in token.children if c.dep_ in subj_deps]
                raw_objects = [c for c in token.children if c.dep_ in obj_deps]

                # include objects reached via prepositions (prep -> pobj)
                for prep in (c for c in token.children if c.dep_ == "prep"):
                    raw_objects.extend([c for c in prep.children if c.dep_ == "pobj"])

                subjects = filter_and_dedupe(raw_subjects)
                objects = filter_and_dedupe(raw_objects)

                # if there are subjects and objects, create triples
                if subjects and objects:
                    for subj in subjects:
                        subj_group = [subj] + list(subj.conjuncts)
                        for s_tok in subj_group:
                            s_norm = normalize_arg(s_tok)
                            if not s_norm:
                                continue
                            s_text = phrase_text(s_norm)
                            kg.add_node(s_text, lemma=s_norm.lemma_, pos=s_norm.pos_)
                            for obj in objects:
                                obj_group = [obj] + list(obj.conjuncts)
                                for o_tok in obj_group:
                                    o_norm = normalize_arg(o_tok)
                                    if not o_norm:
                                        continue
                                    o_text = phrase_text(o_norm)
                                    kg.add_node(o_text, lemma=o_norm.lemma_, pos=o_norm.pos_)
                                    rel = token.lemma_.lower()
                                    kg.add_edge(s_text, o_text, relation=rel, dep=token.dep_)
                                    triples.append((s_text, rel, o_text))

                # also handle verb -> prepositional relations when no direct object
                elif subjects:
                    for subj in subjects:
                        subj_group = [subj] + list(subj.conjuncts)
                        for s_tok in subj_group:
                            s_norm = normalize_arg(s_tok)
                            if not s_norm:
                                continue
                            s_text = phrase_text(s_norm)
                            for prep in (c for c in token.children if c.dep_ == "prep"):
                                for pobj in (c for c in prep.children if c.dep_ == "pobj"):
                                    o_norm = normalize_arg(pobj)
                                    if not o_norm:
                                        continue
                                    o_group = [o_norm] + list(o_norm.conjuncts)
                                    for o_tok in o_group:
                                        o_text = phrase_text(o_tok)
                                        kg.add_node(s_text, lemma=s_norm.lemma_, pos=s_norm.pos_)
                                        kg.add_node(o_text, lemma=o_tok.lemma_, pos=o_tok.pos_)
                                        rel = f"{token.lemma_.lower()}_{prep.text.lower()}"
                                        kg.add_edge(s_text, o_text, relation=rel, dep=token.dep_)
                                        triples.append((s_text, rel, o_text))

    # remove nodes that contain week day names (case-insensitive)
    week_pattern = re.compile(r"\b(?:mon|monday|tue|tuesday|wed|wednesday|thu|thursday|fri|friday|sat|saturday|sun|sunday)\b", flags=re.I)
    to_drop = [n for n in kg.nodes() if week_pattern.search(n)]
    if to_drop:
        for n in to_drop:
            if kg.has_node(n):
                kg.remove_node(n)
        triples = [t for t in triples if t[0] in kg.nodes() and t[2] in kg.nodes()]

    # enforce min_edges constraint (if set)
    if min_edges is not None:
        rem = [n for n, d in kg.degree() if d < int(min_edges)]
        if rem:
            for n in rem:
                kg.remove_node(n)
            triples = [t for t in triples if t[0] in kg.nodes() and t[2] in kg.nodes()]


    # optional: store triples on the graph for later use
    if triples:
        kg.graph.setdefault("triples", []).extend(triples)

    print(kg)
    if triples:
        for triple in triples:
            print(triple)
    return kg


def kg_visualisation(kgs, combined):
    # if nothing to draw, skip
    if combined.number_of_nodes() == 0:
        print("Combined KG is empty, skipping visualization.")
    else:
        plt.figure(figsize=(22, 14))
        pos = nx.spring_layout(combined, k=0.9, seed=42)
        degrees = dict(combined.degree())
        node_sizes = [300 + degrees.get(n, 0) * 150 for n in combined.nodes()]

        # simple palette (extend if you have more languages)
        palette = ["steelblue", "orange", "green", "red", "brown", "cyan", "magenta", "gold", "teal", "gray"]
        sorted_langs = sorted(kgs.keys())
        lang_colors = {lang: palette[i % len(palette)] for i, lang in enumerate(sorted_langs)}

        color_map = []
        for n in combined.nodes():
            sources = combined.nodes[n].get("sources", set())
            if len(sources) == 1:
                color_map.append(lang_colors.get(next(iter(sources)), "gray"))
            else:
                color_map.append("purple")  # present in multiple languages

        nx.draw_networkx_nodes(combined, pos, node_size=node_sizes, node_color=color_map, edgecolors="k")
        nx.draw_networkx_labels(combined, pos, font_size=9)
        nx.draw_networkx_edges(
            combined,
            pos,
            arrowstyle="->",
            arrowsize=12,
            edge_color="gray",
            connectionstyle="arc3,rad=0.08",
            width=1.0,
        )

        # build edge labels
        edge_labels = {}
        for u, v, d in combined.edges(data=True):
            rels = sorted([r for r in d.get("relations", set()) if r])
            edge_labels[(u, v)] = ", ".join(rels) if rels else ""

        if any(edge_labels.values()):
            nx.draw_networkx_edge_labels(combined, pos, edge_labels=edge_labels, font_color="red", font_size=8, label_pos=0.5)

        # legend: one patch per language + one for multiple
        import matplotlib.patches as mpatches
        patches = []
        for lang in sorted_langs:
            patches.append(mpatches.Patch(color=lang_colors[lang], label=lang))
        patches.append(mpatches.Patch(color="purple", label="multiple languages"))
        plt.legend(handles=patches, loc="upper right")

        plt.title("Combined Knowledge Graph (color = single-language node, purple = multiple languages)")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "kg_combined_multi.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved combined KG visualization to {out_path}")
        plt.close()


def to_translate(texts, l):
    translated = []
    if not texts:
        return translated
    
    for t in texts:
        translation = GoogleTranslator(source=l, target='en').translate(text=t)
        translated.append(translation)
    
    return translated

def main():
    today = datetime.utcnow().date()

    # 1-week range (used by fetch_headlines)
    from_date = today.isoformat()
    to_date = (today - timedelta(weeks=1)).isoformat()

    print(f"One-week range: from_date={from_date}, to_date={to_date}")

    all_news = {}
    for lang in considered_langs:
        # build sources for this specific language
        lang_sources = [s['id'] for s in r["sources"] if s['language'] == lang]
        src_param = ",".join(lang_sources)  # newsapi expects comma-separated source ids
        texts = fetch_headlines(QUERY, src_param, NEWSAPI_KEY, from_date, to_date, lang)
        # print(texts)
        if lang != "en":
            translated_non_english = to_translate(texts, lang)
            all_news[lang] = translated_non_english
        else:
            all_news[lang] = texts

    # build KGs for all languages and store in a dict
    kgs = {}
    for lang, texts in all_news.items():
        kgs[lang] = kg_creation(texts)

    # brief summary
    for lang, g in kgs.items():
        print(f"KG for {lang}: nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")

    # combine all KGs into one graph and mark node/edge sources (languages)
    combined = nx.DiGraph()
    for lang, g in kgs.items():
        if not g or g.number_of_nodes() == 0:
            continue
        for n, data in g.nodes(data=True):
            if not combined.has_node(n):
                combined.add_node(n)
                for k, v in (data or {}).items():
                    if k == "sources":
                        continue
                    combined.nodes[n].setdefault(k, v)
            combined.nodes[n].setdefault("sources", set()).add(lang)

        for u, v, data in g.edges(data=True):
            rel = (data or {}).get("relation", "")
            if combined.has_edge(u, v):
                combined[u][v].setdefault("relations", set()).add(rel)
                combined[u][v].setdefault("sources", set()).add(lang)
            else:
                combined.add_edge(u, v, relations=set([rel]) if rel else set(), sources=set([lang]))

    kg_visualisation(kgs, combined)

if __name__ == "__main__":
    main()

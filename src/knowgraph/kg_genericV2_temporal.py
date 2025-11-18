import os
import re
import requests
import networkx as nx
import spacy
import matplotlib.pyplot as plt
from dotenv import load_dotenv, dotenv_values 
from difflib import SequenceMatcher
from newsapi import NewsApiClient
from networkx.algorithms import community as nx_comm
from spacy.pipeline import Sentencizer
import os
from datetime import datetime, timedelta

load_dotenv()
NEWSAPI_KEY = (os.getenv("NEWSAPI_KEY"))

OUTPUT_DIR = "/home/miharc/work/code/event_extraction/src/knowgraph/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

r = newsapi.get_sources()
sources = [s['id'] for s in r["sources"] if s['language'] == "en"]

nlp = spacy.load("en_core_web_sm")

QUERY = "Donald Trump"
# def fetch_headlines(query, sources, api_key, language="en", page_size=10):
def fetch_headlines(query, sources, api_key, from_date, to_date, language="en", page_size=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sources": sources,
        "language": language,
        "pageSize": page_size,
        "sortBy": "relevancy",
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

    kg_creation.min_edges = 2  # optional: nodes must have degree >= min_edges
    min_edges = getattr(kg_creation, "min_edges", None)

    # extract simple SVO triples and add them to the kg
    subj_deps = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
    obj_deps = {"dobj", "dative", "attr", "oprd", "pobj", "obj"}
    triples = []

    for t in texts:
        sentences = [i for i in nlp(t).sents]
        for s in sentences:
            # print(s)
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


def kg_visualisation(kg1, kg2):
    def draw_kg(kg, title, outname):
        if kg is None or kg.number_of_nodes() == 0:
            print(f"{title}: KG is empty, nothing to visualise.")
            return
        plt.figure(figsize=(18, 12))
        pos = nx.spring_layout(kg, k=0.9, seed=42)
        degrees = dict(kg.degree())
        node_sizes = [300 + degrees.get(n, 0) * 150 for n in kg.nodes()]

        nx.draw_networkx_nodes(kg, pos, node_size=node_sizes, node_color="skyblue", edgecolors="k")
        nx.draw_networkx_labels(kg, pos, font_size=9)
        nx.draw_networkx_edges(
            kg,
            pos,
            arrowstyle="->",
            arrowsize=12,
            edge_color="gray",
            connectionstyle="arc3,rad=0.08",
            width=1.0,
        )

        edge_labels = {(u, v): d.get("relation", "") for u, v, d in kg.edges(data=True)}
        if any(edge_labels.values()):
            nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color="red", font_size=8, label_pos=0.5)

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, outname)
        plt.savefig(out_path, dpi=200)
        print(f"Saved KG visualization to {out_path}")
        plt.show()
        plt.close()

    # draw each KG independently
    draw_kg(kg1, "Knowledge Graph - Recent (past week)", "kg_recent.png")
    draw_kg(kg2, "Knowledge Graph - Older (2 weeks ago)", "kg_older.png")

    # combined visualization with color-coding
    if (kg1 is None or kg1.number_of_nodes() == 0) and (kg2 is None or kg2.number_of_nodes() == 0):
        print("Both KGs empty, skipping combined visualisation.")
        return

    combined = nx.DiGraph()

    # add nodes and mark sources
    for g, tag in ((kg1, "recent"), (kg2, "older")):
        if not g:
            continue
        for n, data in g.nodes(data=True):
            # ensure node exists and has a sources set
            if not combined.has_node(n):
                combined.add_node(n)
                # copy node attributes except "sources"
                for k, v in (data or {}).items():
                    if k == "sources":
                        continue
                    combined.nodes[n].setdefault(k, v)
            # always mark the node as coming from this tag
            combined.nodes[n].setdefault("sources", set()).add(tag)

    # add edges and accumulate relations per edge
    for g, tag in ((kg1, "recent"), (kg2, "older")):
        if not g:
            continue
        for u, v, data in g.edges(data=True):
            rel = data.get("relation", "") if data else ""
            if combined.has_edge(u, v):
                combined[u][v].setdefault("relations", set()).add(rel)
                combined[u][v].setdefault("sources", set()).add(tag)
            else:
                combined.add_edge(u, v, relations=set([rel]) if rel != "" else set(), sources=set([tag]))

    # prepare drawing attributes
    plt.figure(figsize=(22, 14))
    pos = nx.spring_layout(combined, k=0.9, seed=42)
    degrees = dict(combined.degree())
    node_sizes = [300 + degrees.get(n, 0) * 150 for n in combined.nodes()]

    color_map = []
    for n in combined.nodes():
        sources = combined.nodes[n].get("sources", set())
        # ensure we compare sets, not rely on identity
        if sources == {"recent"}:
            color_map.append("steelblue")   # recent only
        elif sources == {"older"}:
            color_map.append("orange")      # older only
        else:
            color_map.append("purple")      # present in both

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

    # build edge labels from accumulated relations
    edge_labels = {}
    for u, v, d in combined.edges(data=True):
        rels = sorted([r for r in d.get("relations", set()) if r])
        edge_labels[(u, v)] = ", ".join(rels) if rels else ""

    if any(edge_labels.values()):
        nx.draw_networkx_edge_labels(combined, pos, edge_labels=edge_labels, font_color="red", font_size=8, label_pos=0.5)

    # legend (create simple patch handles)
    import matplotlib.patches as mpatches
    recent_patch = mpatches.Patch(color="steelblue", label="Recent (past week)")
    older_patch = mpatches.Patch(color="orange", label="Older (2 weeks ago)")
    both_patch = mpatches.Patch(color="purple", label="Both")
    plt.legend(handles=[recent_patch, older_patch, both_patch], loc="upper right")

    plt.title("Combined Knowledge Graph (color: recent / older / both)")
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "kg_combined.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved combined KG visualization to {out_path}")
    # plt.show()
    plt.close()


def main():
    today = datetime.utcnow().date()

    # 1-week range (used by fetch_headlines)
    from_date = today.isoformat()
    to_date = (today - timedelta(weeks=1)).isoformat()

    # 2-week range (alternative)
    from_date_2 = (today - timedelta(weeks=1)).isoformat()
    to_date_2 = (today - timedelta(weeks=2)).isoformat()

    print(f"One-week range: from_date={from_date}, to_date={to_date}")
    
    texts = fetch_headlines(QUERY, sources, NEWSAPI_KEY, from_date, to_date)
    # print(texts)
    # print()
    kg1 = kg_creation(texts)

    print(f"Two-week range: from_date={from_date_2}, to_date={to_date_2}")
    texts = fetch_headlines(QUERY, sources, NEWSAPI_KEY, from_date_2, to_date_2)
    # print(texts)
    kg2 = kg_creation(texts)
    kg_visualisation(kg1, kg2)

if __name__ == "__main__":
    main()

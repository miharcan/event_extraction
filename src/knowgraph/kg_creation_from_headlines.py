import os
import re
import requests
import networkx as nx
import spacy
import matplotlib.pyplot as plt
from dotenv import load_dotenv, dotenv_values 
from difflib import SequenceMatcher
import copy
import math
from newsapi import NewsApiClient

load_dotenv()
NEWSAPI_KEY = (os.getenv("NEWSAPI_KEY"))

# ---- simple configuration ----
QUERY = "Donald Trump"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

r = newsapi.get_sources()
sources = [s['id'] for s in r["sources"] if s['language'] == "en"]

nlp = spacy.load("en_core_web_sm")

# ---- fetching headlines  ----
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
        text = re.sub(r"\[.*?\]", "", text).strip()
        texts.append(text)
    return texts

# small helper to build a short phrase for a token (compounds + token)
def phrase_for_token(tok):
    parts = []
    for child in tok.lefts:
        if child.dep_ in ("compound", "amod"): #, "det"):
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
                        if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN"): #, "PRON"):
                            subj = phrase_for_token(ch)
                        if ch.dep_ in ("dobj", "obj", "pobj") and ch.pos_ in ("NOUN", "PROPN"): #, "PRON"):
                            obj = phrase_for_token(ch)
                    ####also allow prepositional object as object if no direct object
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
def draw_and_save_kg(kg, out_path, name):
    out_path = os.path.join(OUTPUT_DIR, name)

    n = kg.number_of_nodes()
    # make figure size grow with number of nodes (bounded)
    figsize = (min(24, 6 + n * 0.25), min(18, 4 + n * 0.18))
    plt.figure(figsize=figsize)

    # choose a larger 'k' for spring_layout to spread nodes further.
    # scale and iterations increased for more stable, spread-out layouts.
    try:
        if n > 1:
            # k roughly proportional to sqrt(n); tune multiplier for more/less spacing
            k = 1.5 * math.sqrt(n)
            pos = nx.spring_layout(kg, seed=42, k=k, iterations=200, scale=2.0)
        else:
            # single node
            pos = {list(kg.nodes())[0]: (0, 0)}
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

    # reduce node size slightly when many nodes to avoid overlap
    node_size = int(max(150, 800 - n * 8))

    nx.draw_networkx_nodes(kg, pos, node_color=node_colors, node_size=node_size, alpha=0.95)
    nx.draw_networkx_labels(kg, pos, font_size=9)
    nx.draw_networkx_edges(kg, pos, arrowstyle="->", arrowsize=10, edge_color="#444444", width=1.0)
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in kg.edges(data=True)}
    if edge_labels:
        nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color="gray", font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved KG image: {out_path}")
    print(kg)
    with open(f"{name}.info", "w") as file:
        file.write(f"Built KG with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges.\n")
        for n, d in list(kg.nodes(data=True)):
            file.write(f"{n}, {d}\n")
        for u, v, d in list(kg.edges(data=True)):
            file.write(f"{u} -[{d.get('relation')}]-> {v}\n")


def simple_entity_linking(kg, threshold=0.8, verbose=False):
    # only consider string node ids (guard against non-string node ids)
    all_nodes = [n for n in kg.nodes() if isinstance(n, str)]
    nodes_by_len = {}
    for n in all_nodes:
        nodes_by_len.setdefault(len(n), []).append(n)

    # store/initialize merged map once
    merged = kg.graph.setdefault("merged", {})

    # sort nodes by length descending so we prefer longer 'keep' candidates
    sorted_nodes = sorted(all_nodes, key=lambda s: len(s), reverse=True)

    # compare each pair once: for i, compare with j > i
    for i, n in enumerate(sorted_nodes):
        if n not in kg:
            continue
        for n2 in sorted_nodes[i + 1 :]:
            if n2 not in kg:
                continue
            if len(n) < len(n2):  # ensure outer length >= inner length
                continue
            if n == n2:
                continue
            
            sim = SequenceMatcher(None, n.lower(), n2.lower()).ratio()
            if sim < threshold:
                continue

            keep, remove = n, n2
            if keep not in kg or remove not in kg or keep == remove:
                continue

            # incoming edges -> keep (merge/overwrite attributes if necessary)
            for pred, _, data in list(kg.in_edges(remove, data=True)):
                if pred == keep:
                    continue
                if kg.has_edge(pred, keep):
                    kg[pred][keep].update(data)
                else:
                    kg.add_edge(pred, keep, **data)

            # outgoing edges from remove -> keep
            for _, succ, data in list(kg.out_edges(remove, data=True)):
                if succ == keep:
                    continue
                if kg.has_edge(keep, succ):
                    kg[keep][succ].update(data)
                else:
                    kg.add_edge(keep, succ, **data)

            # merge node attributes: keep wins, but copy missing attrs from remove
            attrs_keep = dict(kg.nodes[keep])
            attrs_remove = dict(kg.nodes[remove])

            # special handling for "type" key: concatenate if different
            t_keep = attrs_keep.get("type")
            t_remove = attrs_remove.get("type")
            if t_keep and t_remove and t_keep != t_remove:
                attrs_keep["type"] = f"{t_keep}|{t_remove}"
            elif not t_keep and t_remove:
                attrs_keep["type"] = t_remove

            # copy any attributes from remove that keep does not have
            for k_attr, v_attr in attrs_remove.items():
                if k_attr not in attrs_keep:
                    attrs_keep[k_attr] = v_attr

            kg.nodes[keep].update(attrs_keep)

            # record merge and remove node
            merged[remove] = {"merged_into": keep, "sim": round(sim, 3)}
            try:
                kg.remove_node(remove)
            except Exception as ex:
                # don't silently swallow errors - keepable for debugging or real handling
                if verbose:
                    print(f"Failed to remove node {remove}: {ex}")
            kg.graph["merged"] = merged
            print(f"Merged '{remove}' -> '{keep}' (sim={sim:.2f})")

    return merged

# ---- main flow ----
def main():
    texts = fetch_headlines(QUERY, ", ".join(sources), NEWSAPI_KEY)
    if not texts:
        print("No articles fetched.")
        return
    kg = build_kg_from_texts(texts)
    
    kg_linked = kg.copy()
    entity_linked = simple_entity_linking(kg_linked)

    if entity_linked:
        canonical_map = kg_linked.graph.setdefault("canonical_map", {})
        for removed, info in entity_linked.items():
            keep = info.get("merged_into")
            if not keep:
                continue
            canonical_map[removed] = keep
            if keep in kg_linked:
                aliases = kg_linked.nodes[keep].setdefault("aliases", [])
                if removed not in aliases:
                    aliases.append(removed)
        kg_linked.graph["canonical_map"] = canonical_map
    else:
        kg_linked.graph.setdefault("canonical_map", {})
    draw_and_save_kg(kg, OUTPUT_DIR, "original_kg.png")
    draw_and_save_kg(kg_linked, OUTPUT_DIR, "linked_kg.png")

if __name__ == "__main__":
    main()

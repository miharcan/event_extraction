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
from networkx.algorithms import community as nx_comm
import textwrap
from spacy.pipeline import Sentencizer
import os

load_dotenv()
NEWSAPI_KEY = (os.getenv("NEWSAPI_KEY"))


# SOURCES = "bbc-news,abc-news,al-jazeera-english,associated-press, bbc-sport"
OUTPUT_DIR = "/home/miharc/work/code/event_extraction/src/knowgraph/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

r = newsapi.get_sources()
sources = [s['id'] for s in r["sources"] if s['language'] == "en"]

nlp = spacy.load("en_core_web_sm")

QUERY = ""
# GET https://newsapi.org/v2/top-headlines?country=us&apiKey=34d1936418414b60b6f11621ad094523
# ---- fetching headlines  ----
# def fetch_headlines(query, sources, api_key, language="en", page_size=10):
def fetch_headlines(query, sources, api_key, language="en", page_size=100):
    url = "https://newsapi.org/v2/top-headlines"
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
        # text = f"{desc}. {cont}"
        text = cont
        text = re.sub(r"\[.*?\]", "", text).strip()
        texts.append(text)
    return texts

def kg_creation(texts):
    kg = nx.DiGraph()

    # extract simple SVO triples and add them to the kg
    subj_deps = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
    obj_deps = {"dobj", "dative", "attr", "oprd", "pobj", "obj"}
    triples = []

    for t in texts:
        sentences = [i for i in nlp(t).sents]
        for s in sentences:
            print(s)
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

    # optional: store triples on the graph for later use
    if triples:
        kg.graph.setdefault("triples", []).extend(triples)

    print(kg)
    if triples:
        for triple in triples:
            print(triple)
    return kg


def kg_visualisation(kg):
    if kg is None or kg.number_of_nodes() == 0:
        print("KG is empty, nothing to visualise.")
        return

    plt.figure(figsize=(24, 16))
    # layout
    pos = nx.spring_layout(kg, k=0.9, seed=42)

    # node sizes by degree
    degrees = dict(kg.degree())
    node_sizes = [300 + degrees.get(n, 0) * 150 for n in kg.nodes()]

    # draw nodes and labels
    nx.draw_networkx_nodes(kg, pos, node_size=node_sizes, node_color="skyblue", edgecolors="k")
    nx.draw_networkx_labels(kg, pos, font_size=9)

    # draw directed edges
    nx.draw_networkx_edges(
        kg,
        pos,
        arrowstyle="->",
        arrowsize=12,
        edge_color="gray",
        connectionstyle="arc3,rad=0.08",
        width=1.0,
    )

    # draw edge labels from the 'relation' attribute if present
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in kg.edges(data=True)}
    if any(edge_labels.values()):
        nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color="red", font_size=8, label_pos=0.5)

    plt.title("Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "kg_generic.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved KG visualization to {out_path}")
    plt.show()


def main():
    # texts = fetch_headlines(QUERY, sources, NEWSAPI_KEY)
    # texts = ['An American man and his teenage son have died after they were stung by a swarm of wasps while ziplining in Laos last month BANGKOK -- An American man and his teenage son died last month after they were swarmed by wasps while ziplining at an adventure camp in Laos and stung many dozens of times, a hospital official said T…', 'The British military says that attackers firing machine guns and rocket-propelled grenades boarded a ship off the coast of Somalia DUBAI, United Arab Emirates -- Attackers firing machine guns and rocket-propelled grenades boarded a tanker carrying gasoline off the coast of Somalia on Thursday, authorities said, likely the latest…', 'Flight capacity will be reduced by 10% at 40 major airports. The Federal Aviation Administration will reduce flight capacity by 10% at 40 major airports across the country, officials announced during a press conference on Wednesday.\r\nThe decision could cut tho…', '"Last night was a great night for America," Schumer said. Frustrations are boiling over as a confrontation unfolded between a rank-and-file Democrat and Speaker Mike Johnson on the steps of the House of Representatives.\r\nPennsylvania Democratic Rep. Chrissy…', 'The third skier remains missing. Nearly eight months after the deadliest U.S. avalanche in two years, authorities say they have now recovered the bodies of two of the three men who were buried in up to 100 feet of snow.\r\nThe second …', 'The death toll may rise as officials search for additional victims. At least 12 people are dead after the left engine separated from a UPS plane when departing Louisville Muhammad Ali International Airport in Kentucky on Tuesday and it crashed in a ball of flames, au…', 'Abby Zwerner sued for $40 million over the 2023 shooting. A Virginia jury heard closing arguments Wednesday in a $40 million civil case in which they will decide whether an assistant principal acted with gross negligence when a then-6-year-old student shot …', "Duwaji appeared with Mamdani after he secured his election win on Tuesday night. After Zohran Mamdani successfully completed his historic bid for mayor of the country's largest city on Tuesday night, he was joined onstage by his wife Rama Duwaji.\r\nIn his speech, Mamdani thanked D…", 'The bread-and-butter advantage among Democrats marked a sharp turnabout. A Navy veteran, an ex-member of Congress and a self-avowed democratic socialistwon resounding victories in high-profile races on Tuesday all carrying the banner of the Democratic party and each addre…', 'Two men were also arrested last week in the alleged Michigan plot. Two New Jersey teenagers have been arrested in connection with an alleged ISIS-inspired Halloween attack in Michigan that the FBI announced it had thwarted last week, law enforcement sources told ABC…']
    texts = ['A new customs control facility and border control post has been officially opened at Rosslare Europort in Co Wexford., Referred to as Terminal 7 and built at a cost of close to €230 million, it is one of the biggest capital infrastructure projects of its kind in Ireland., A significant amount of the construction cost came from the EU\'s Brexit Adjustment Reserve (BAR)., Rosslare Europort has seen a more than six-fold increase in direct European sailings since Brexit, with many freight and delivery companies shifting from using the UK landbridge to more direct sea routes with the European Union., The major infrastructure upgrades reflect this growth, as well as the new customs requirements that come with the port becoming a major gateway between Ireland and mainland Europe., Mourners at Sr Stanislaus Kennedy\'s funeral have heard how a "truly great soul has passed from this world"., In his homily, Father Richard Hendrick told the congregation the charity and social justice organisations she founded will remain as a "testimony to the legacy of this powerhouse of good who swept into so many lives"., The Capuchin Franciscan friar and author, known as Brother Richard, led the service at the Church of the Sacred Heart in Dublin., He described how the funeral mass was filled with "real loss and grief" but also "filled with the hope that burned in the heart of Sister Stan her whole life"., He also said Ireland has been "blessed by women who have heard the gospel as the Good News to the poor that it truly is" adding "Sister Stan followed in their footsteps"., In her welcoming remarks, Sr Una of the Religious Sisters of Charity said Sr Stan had planned much of her funeral, adding she had "very definite ideas about how we should celebrate her passing"., Infant remains have been recovered from the site of the former mother-and-baby home in Tuam, Co Galway, as excavation work there continues., However, at this stage, it has not been determined if the bones date from the period during which the institution operated, between 1925 and 1961., Radiocarbon dating is being carried out to determine the \'era of origin\' of seven sets of remains that were recovered from part of the site in recent weeks. This is expected to take a number of months to complete., In its latest update, the Office of the Director of Authorised Intervention in Tuam (ODAIT) said a further two sets of remains were found in a separate location over the last four weeks., These date from the time when a workhouse operated, during the mid 19th to early 20th Century., The ODAIT has previously cautioned that the multiple uses of the site over the last 200 years would complicate the excavation task., At various times it served as a workhouse, a military barracks and a mother-and-baby home., The infant skeletal remains were recovered from an area adjacent to an underground vaulted structure indicated on workhouse plans., Ireland captain Caelan Doris feels stronger mentally and physically as he bids to put his career back on track following the agony of missing this summer\'s British and Irish Lions tour., The 27-year-old will make his first start in six months in Saturday’s autumn international against Japan (12.40pm) after returning from a shoulder injury as a replacement in last weekend’s 26-13 loss to New Zealand in Chicago., Doris was touted as a potential Lions captain for the recent series in Australia before undergoing surgery on a rotator cuff issue sustained in Leinster’s Champions Cup defeat to Northampton on 3 May., His lengthy period of rehabilitation included reconditioning his body, attending a wellness retreat in California, travelling with friends and avoiding his phone during a digital detox., "I haven’t had shoulder injuries in the last number of years, since school really, so it was an unfamiliar feeling – I knew something wasn’t right," said Doris., "I was gutted initially, obviously, but I really felt it allowed the emotion to come through and as a result process it., "I was then able to see the positives in the situation and move on quite quickly and frame it in a positive way and as a, hopefully, halfway point in my career, and reset and do things that I wouldn’t have been able to do had I been playing rugby., Ireland face Japan on Saturday in the early afternoon, and while it\'s an encounter we\'re absolutely expected to win it, it\'s important in the context of the shortcomings last week at the set-piece, in our attack and in our kicking game., We also have to be mindful of what\'s coming down the line over the next two weeks. Joe Schmidt\'s Australia are here on Saturday week, and South Africa, the number one team in the world, the following week., It\'s a huge game from that point of view., It was obvious that Andy Farrell was going to make changes, and he\'s made eight in total, with four up front and four in the backline., It\'s still a very balanced side, mixed with experience and opportunities for other players, including Tom Farrell., Farrell will win his first cap at 32, and last week Stuart McCloskey was brought back into the midfield at 33., Governments heading to the UN COP30 climate summit in Brazil are bracing for the possibility that the Trump administration may seek to disrupt negotiations at the event, even without any US officials showing up., The White House has said it will not send high-level officials to the annual conference, noting that President Donald Trump made his views clear at the UN General Assembly in September when he described climate change as the world\'s "greatest con job"., However, the US retains the option to send negotiators at any point during the 10 to 21 November COP30 talks, ahead of the country formally exiting the international Paris climate agreement in January., Three European officials told Reuters the EU has been preparing for multiple scenarios at COP30, including the US skipping it entirely, actively participating and seeking to block deals, or staging sideline events to denounce climate policies., Ahead of COP30, the UN has said the world will fail to meet its main climate change target of limiting the rise in global temperatures to 1.5C above pre-industrial levels., For years, climate campaigners have said countries need to throw the kitchen sink at tackling climate change., So as hopes of keeping the Paris Agreement\'s key target drains away, we\'re using a kitchen sink to explain what has gone wrong., Shane Lowry\'s challenge for a second Abu Dhabi Championship was dented by back-to-back closing bogeys as he lies three strokes off the leaders at the halfway point in Yas Island., Lowry - who won the tournament in 2019 - carded a second round 69 to sit on 11 under, with playing partner Tommy Fleetwood and Aaron Rai sharing top spot on the leaderboard., The Offaly man had tied the lead after 13 holes following his birdie at the short 13th hole. Prior to that, he had registered birdies at the second and 10th holes., His push was turbo-charged by an eagle at the par-five 11th, despite appearing to thin his approach shot, which Lowry himself labelled "the worst best shot of all-time" before the ball came to a halt 11ft from the cup., The world\'s richest man has been handed a chance to become history\'s first trillionaire., Elon Musk won a shareholder vote on Thursday that would give the Tesla CEO stock worth one trillion dollars, around €865 billion, if he hits certain performance targets over the next decade., The vote followed weeks of debate over his management record at the electric car maker and whether anyone deserved such unprecedented pay, drawing heated commentary from small investors to giant pension funds and even the pope., In the end, more than 75% of voters approved the plan as shareholders gathered in Austin, Texas, for their annual meeting., "Fantastic group of shareholders," Mr Musk said after the final vote was tallied, adding "Hang on to your Tesla stock"., The vote is a resounding victory for Mr Musk showing investors still have faith in him as Tesla struggles with plunging sales, market share and profits in no small part due to himself., Three boys have been arrested in connection with an investigation into serious public disorder in Saggart last month., The unrest occurred during a protest outside an accommodation centre for International Protection applicants at Citywest in Co Dublin on 21 October, with further disorder on 22 October., Gardaí conducted additional searches today as part of the investigation and arrested three boys., The arrests bring to 36 the total number of people arrested as part of the investigation., Gardaí continue to appeal for anyone with information on people involved in the disorder to contact them at Clondalkin Garda Station on 01 6667600, or any member of An Garda Síochána at any garda station., Members of the public can provide information confidentially to An Garda Síochána by contacting the Garda Confidential Line 1800 666 111, gardaí said., A celebration of one of the Yeats Country\'s most distinguished but forgotten sons will take place in Sligo City Hall this evening., William Bourke Cockran, who emigrated to the US aged 17, was a six-term congressman, a renowned orator, lawyer, a confidante to three US presidents, a mentor to Winston Churchill and a key player in Irish and American politics., Keville Burns of the Sligo Field Club said Cockran was "an architect in Ireland\'s new path to freedom" and one of the leading US Catholic laymen in the early 20th century., Born in Carrowkeel in Co Sligo in 1854, Cockran became the first person to receive the Freedom of Sligo Borough in 1903, but his name was never etched in City Hall., However, this will change as his name is to join others who were bestowed with the same honour ahead of the opening of an exhibition and conference on his life in the historic building., The first person convicted and jailed in Ireland for online posts that jeopardised the anonymity of asylum seekers has had a "priority" appeal date set in 2026., Paul Nolan, 37, from Mount Eagle Square, Leopardstown, Dublin, stood outside the IPAS centre in Tallaght on two days in August last year, questioning residents, including teenagers, a young woman, and three men, Dublin District Court heard in September., In that hearing, Judge John Hughes noted this was the first prosecution under the relevant law, which carries a possible 12-month prison term., Nolan received a ten-month sentence, with the final three months suspended if he completes probation, anger management counselling, stays away from IPAS centres for two years, and removes the videos., Father of three Nolan told migrants "in Ireland, you have no right to privacy," and posted videos of his interactions on YouTube., However, shortly after being jailed, he lodged an appeal to overturn his conviction and was released., Three boys have been arrested in connection with an investigation into serious public disorder in Saggart last month., The unrest occurred during a protest outside an accommodation centre for International Protection applicants at Citywest in Co Dublin on 21 October, with further disorder on 22 October., Gardaí conducted additional searches today as part of the investigation and arrested three boys., The arrests bring to 36 the total number of people arrested as part of the investigation., Gardaí continue to appeal for anyone with information on people involved in the disorder to contact them at Clondalkin Garda Station on 01 6667600, or any member of An Garda Síochána at any garda station., Members of the public can provide information confidentially to An Garda Síochána by contacting the Garda Confidential Line 1800 666 111, gardaí said., One of the subplots around Sunday\'s Sports Direct FAI Cup final is that a Shamrock Rovers victory would secure European football next season for their bitter rivals Bohemians., Bohs fans might be cheering them on through gritted teeth, if such a thing is possible., Of course it is also worth pointing out that if the Hoops lift the trophy at Aviva Stadium, Derry City fans will also have cause to celebrate, as it would guarantee their place in the Europa League qualifiers next summer and the extra prize pot that would accrue., It perhaps is no surprise that a Shamrock Rovers player would be asked about the scenario that would play out if they were to complete the league/cup double.']
    kg = kg_creation(texts)
    kg_visualisation(kg)

if __name__ == "__main__":
    main()

import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import pandas as pd
import numpy as np

#python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


headlines = [
    "Massive explosion rocks Beirut port area",
    "Thousands protest government corruption in the capital of United States",
    "Earthquake of magnitude 6.7 hits Japan coast with massive explosion",
    "Blast damages refinery in Kuwait",
    "Demonstrators rally for justice downtown"
]

event_keywords = {
    "explosion": "Disaster",
    "protest": "Civil Unrest",
    "earthquake": "Natural Disaster"
}

proto_text = list(event_keywords.keys())
proto_labels = list(event_keywords.values())

vectoriser = TfidfVectorizer()
tfidf_matrix = vectoriser.fit_transform(proto_text)

def detect_event(text, tfidf_matrix, vectoriser, labels, threshold=0.25):
    vec = vectoriser.transform([text.lower()])
    sim = cosine_similarity(vec, tfidf_matrix)[0]
    indx = np.argmax(sim)
    if (sim[indx] > threshold):
        return labels[indx], sim[indx]
    return "New event:", sim[indx] 
    


def extract_events_semantic(texts:List[str]):
    results = []
    for text in texts:
        event, score = detect_event(text, tfidf_matrix, vectoriser, proto_labels)
        doc = nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        results.append({
            "headline": text,
            "event": event,
            "similarity": round(score, 3),
            "locations": list(set(locations))
        })
    return results

print(extract_events_semantic(headlines))



def extract_events_string_match(texts:List[str], event_dict: Dict[str, str]) -> List[Dict]:
    results = []
    for text in texts:
        events = []
        for word, label in event_dict.items():
            if re.search(rf'\b{word}\b', text.lower()):
                events.append(label)
        
        doc = nlp(text)
        propn = []
        for ent in doc.ents:
            if ent.label_ == "GPE":
                propn.append(ent.text)

        events = list(set(events))
        propn = list(set(propn))
        results.append({"headline": text, "events":events, "entities":propn})
    return results

# print(extract_events_string_match(headlines, event_keywords))

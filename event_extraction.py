import spacy

#python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


headlines = [
    "Massive explosion rocks Beirut port area",
    "Thousands protest government corruption in the capital of United States",
    "Earthquake of magnitude 6.7 hits Japan coast with massive explosion"
]

event_keywords = {
    "explosion": "Disaster",
    "protest": "Civil Unrest",
    "earthquake": "Natural Disaster"
}


def extract_events(texts, event_dict):
    results = []
    for text in texts:
        events = []
        for word, label in event_dict.items():
            if word in text.lower():
                events.append(label)
        
        doc = nlp(text)
        propn = []
        for ent in doc.ents:
            if ent.label_ == "GPE":
                propn.append(ent.text)

        results.append({"headline": text, "events":events, "propns":propn})
    return results

print(extract_events(headlines, event_keywords))

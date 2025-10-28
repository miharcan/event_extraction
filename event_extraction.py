headlines = [
    "Massive explosion rocks Beirut port area",
    "Thousands protest government corruption in the capital",
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
                events.append({"event":label})
        if events != False:
            results.append({"headline": text, "events":events})
    return results

print(extract_events(headlines, event_keywords))

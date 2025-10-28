from src.event_extraction.event_extraction import extract_events_semantic

def test_basic_extraction():
    headlines = ["Explosion in Madrid"]
    results = extract_events_semantic(headlines)
    assert results[0]["event"] != "New Event"

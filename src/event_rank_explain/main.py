import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import re
import numpy as np
import spacy

#python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Sample humanitarian event texts
data = [
    {"id": 1, "text": "Armed clashes reported between groups in Khartoum, Sudan."},
    {"id": 2, "text": "Floods displaced thousands in Mogadishu, Somalia."},
    {"id": 3, "text": "Food aid distributed to refugee camps near Addis Ababa."},
    {"id": 4, "text": "Ceasefire negotiations begin between rebel and government forces in Sudan."},
    {"id": 5, "text": "Medical supplies shortage reported after conflict escalation in Darfur."}
]

query = ["conflict events in Sudan"]

df = pd.DataFrame(data)

model = SentenceTransformer('all-MiniLM-L6-v2')
data_embeddings = model.encode(df['text'], normalize_embeddings = True)
query_embedding = model.encode(query, normalize_embeddings = True)

df['cosine score data query'] = cosine_similarity(query_embedding, data_embeddings).flatten()
# print(df.sort_values(by="score", ascending=False))


##Explaination
explainer = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
def explain_similarity(query, text):
    prompt = f"Explain why the following text is relevant to '{query}':\n\n{text}\n\nExplanation:"
    return explainer(prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']

explain_lst = []
for i in range(len(df)):
    explain = explain_similarity(query[0], df.loc[i, "text"])
    explain = re.sub(r"\n", " ", explain)
    explain = re.sub(r"  ", " ", explain)
    explain = re.search(r'Explanation:(.+)', explain).group(1)
    explain_lst.append(explain)
df["explanation"] = explain_lst

expl_embs = model.encode(explain_lst, normalize_embeddings=True)
df["cosine similarity query explanation"] = cosine_similarity(query_embedding,expl_embs).flatten()

#Entity Extraction
events = []
entities = []
for text in df["text"]:
    if "conflict" in text:
        events.append("conflict")
    else:
        events.append("NA")
    doc = nlp(text)
    ent_tmp = []
    for ent in doc.ents:
        if ent.label_ == "GPE":
            ent_tmp.append(ent.text)
    if not ent_tmp:
        ent_tmp.append(["NA"])
    entities.append(ent_tmp)

df["events"] = events
df["entities"] = entities
print(df)
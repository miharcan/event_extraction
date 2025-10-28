import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

kb = {
    "Paris_City": np.random.rand(300),
    "Paris_Hilton": np.random.rand(300),
    "Hilton_Hotels": np.random.rand(300)
}

mention_vec = np.random.rand(300)

def link_entity(mention_vec, kb):
    sims = {entity: cosine_similarity([mention_vec], [vec])[0][0] for entity, vec in kb.items()}
    return max(sims, key=sims.get)

print(link_entity(mention_vec, kb))

import networkx as nx

events = [
    {"event": "Explosion", "location": "Beirut", "time": "2020-08-04"},
    {"event": "Protest", "location": "Beirut", "time": "2023-05-01"}
]

G = nx.DiGraph()
for e in events:
    G.add_node(e["event"], type="event")
    G.add_node(e["location"], type="location")
    
    G.add_edge(e["event"], e["location"], relation="occurred_in")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.savefig('kg.png', bbox_inches='tight', dpi=300)
plt.close()

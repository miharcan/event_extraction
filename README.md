# Event Extraction & Knowledge Graph Builder

This project demonstrates lightweight NLP methods for detecting events, linking entities, and visualizing relationships from text or news data.

---

## 🚀 Overview

The codebase combines three main components:

1. **Event Extraction** – Detects and classifies events from short texts or headlines using TF-IDF similarity and keyword matching.
2. **Semantic Search & Explanation** – Uses transformer embeddings (Sentence-BERT) and GPT-Neo to find and explain relationships between humanitarian event texts.
3. **Knowledge Graph Generation** – Builds and visualizes a directed graph of subjects, verbs, and objects extracted from live news articles using the NewsAPI.

---

## Features
- Named Entity Recognition (NER) via **spaCy**
- Semantic similarity with **SentenceTransformers**
- Text explanation using **GPT-Neo (EleutherAI)**
- Event and entity linking via **string similarity**
- Automatic **Knowledge Graph** creation and visualization with **NetworkX** + **Matplotlib**

---

# Knowledge Graph Example

The graph visualizes extracted events, entities, and their relationships.
![Knowledge Graph](https://raw.githubusercontent.com/miharcan/event_extraction/main/src/knowgraph/output/linked_kg.png)

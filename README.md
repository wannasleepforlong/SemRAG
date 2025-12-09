# SemRAG: Semantic Knowledge-Augmented Retrieval-Augmented Generation

SemRAG is a **Semantic Knowledge-Augmented RAG system** designed to improve question-answering by integrating semantic chunking, hybrid retrieval, lightweight knowledge-graph construction, and graph-aware reasoning. The goal is to address limitations of traditional RAG systems such as poor chunk boundaries, weak retrieval signals, and missing conceptual relationships.

SemRAG provides a practical alternative to GraphRAG by focusing on semantic coherence and efficient retrieval, while avoiding the computational overhead of LLM-based graph extraction and summarization.
[Arxiv](https://arxiv.org/abs/2507.21110) [Researchgate](https://www.researchgate.net/publication/394100139_SemRAG_Semantic_Knowledge-Augmented_RAG_for_Improved_Question-Answering)

---

## Features

### 1. Semantic Chunking
- Chunks are generated based on **embedding similarity** rather than fixed token limits
- Consecutive sentences are evaluated using cosine distance
- Boundaries are placed only where semantic discontinuity is detected

### 2. Hybrid Retrieval
SemRAG combines multiple retrieval signals:
- **Vector search** using Chroma and HuggingFace embeddings
- **Keyword search** using BM25
- **Graph-aware expansion** using entity and relation information

### 3. Knowledge Graph Construction
The system extracts:
- **Entities** using spaCy NER
- **Relations** using SVO (Subject-Verb-Object) dependency parsing
- A **directed multigraph** of entities and relations
- **Community identifiers** via Louvain community detection

The graph provides relational context that enhances retrieval relevance.

### 4. Graph-Augmented Context Expansion
Query-time retrieval includes:
- Vector-retrieved chunks
- BM25 keyword matches
- Graph-derived evidence such as relation edges and community neighbors

This produces richer and more contextual evidence for the final answer.

### 5. Grounded LLM Answering
- An LLM is used strictly for **answer synthesis**
- A constrained prompt forces the model to answer only from retrieved context and to avoid unsupported statements
- This reduces hallucinations and improves factual grounding

---

## Pipeline Overview

1. **Extract** raw text from PDF
2. **Generate** semantic chunks based on sentence embeddings
3. **Index** chunks using:
   - Chroma (vector search)
   - BM25 (keyword search)
4. **Extract** entities and relations from chunks
5. **Build** a directed multigraph using NetworkX
6. **Detect** semantic communities using the Louvain algorithm
7. **At query time:**
   - Perform hybrid retrieval (vector + BM25)
   - Retrieve graph-related context (relations and community neighbors)
   - Rerank all evidence using a combined semantic and keyword scoring model
   - Feed top-ranked context into the LLM with a constrained prompt
8. **Produce** a final, grounded answer

---

# Cost-Efficient RAG

## Retrieval-Augmented Generation Using Local LLMs

A cost-efficient Retrieval-Augmented Generation (RAG) system that answers natural-language questions using an indexed collection of Python documentation.

The system combines semantic search with a locally hosted Large Language Model to generate answers grounded in retrieved documentation. Instead of relying on paid cloud APIs, the project uses **Ollama** to run the LLM locally.

---

## Overview

The system follows a complete RAG pipeline:

```text
                 User Question
                       │
                       ▼
              Query Embedding
                       │
                       ▼
              Semantic Retrieval
                       │
                       ▼
                  ChromaDB
                       │
                       ▼
            Top-K Relevant Chunks
                       │
                       ▼
               Context Building
                       │
                       ▼
                Local LLM
                 (Ollama)
                       │
                       ▼
              Grounded Answer
                       │
                       ▼
          Answer + Citations + Metrics

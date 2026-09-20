# ⚖️ ClauseLens

### Contract Intelligence using Clause-Aware Hybrid Retrieval

ClauseLens is an evaluated Retrieval-Augmented Generation (RAG) system for answering questions about legal contracts.

Instead of sending an entire contract directly to an LLM, ClauseLens first retrieves the most relevant contract clauses using lexical and semantic search, combines the retrieval results using Reciprocal Rank Fusion (RRF), and then provides the retrieved evidence to an LLM for grounded answer generation.

The project focuses on **retrieval quality and evaluation**, not just building a basic "chat with PDF" application.

---

## 🎯 Project Objective

Legal contracts are long, highly structured documents containing important information across different clauses and sections.

A simple LLM-based question-answering system can suffer from:

- Long document context
- Irrelevant retrieved information
- Lexical mismatch between questions and contracts
- Semantic mismatch
- Retrieval failures
- Hallucinated answers

ClauseLens addresses these challenges by experimenting with:

1. Different document chunking strategies
2. Lexical retrieval using BM25
3. Semantic retrieval using embeddings
4. Vector search using FAISS
5. Hybrid retrieval using Reciprocal Rank Fusion
6. Quantitative retrieval evaluation
7. Category-level failure analysis
8. Grounded LLM generation

---

# 🏗️ System Architecture

```text
                    CUAD Contracts
                          │
                          ▼
                  Data Ingestion
                          │
                          ▼
                 Contract Processing
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       Fixed Chunking          Clause-Aware Chunking
              │                       │
              ▼                       ▼
            BM25              Sentence Embeddings
              │                       │
              │                     FAISS
              │                       │
              └───────────┬───────────┘
                          ▼
                  Hybrid Retrieval
                       using RRF
                          │
                          ▼
                  Retrieval Evaluation
                   Hit@K + MRR
                          │
                          ▼
                   Relevant Clauses
                          │
                          ▼
                         LLM
                          │
                          ▼
                  Grounded Answer
                          │
                          ▼
                    Streamlit UI



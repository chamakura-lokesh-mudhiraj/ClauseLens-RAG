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


📚 Dataset

ClauseLens uses the CUAD (Contract Understanding Atticus Dataset).

CUAD is a legal contract dataset containing expert annotations for contract review questions.

Dataset statistics used in this project
510 contracts
41 question categories
20,910 total questions
6,702 answerable questions
14,208 impossible/unanswerable questions

Example question categories include:

Parties
Agreement Date
Effective Date
Expiration Date
Renewal Term
Governing Law
Non-Compete
Exclusivity
Termination For Convenience
Change Of Control
Anti-Assignment
License Grant
IP Ownership
Audit Rights
Cap On Liability
Uncapped Liability
Insurance

The expert answer spans provided by CUAD are used as retrieval ground truth.

🔍 Retrieval Problem

Given:

Contract
+
Question

the retrieval system must find the chunk containing the relevant answer.

For example:

Question:
"What is the governing law?"

                    ↓

Retrieve relevant contract clause

                    ↓

"This Agreement shall be governed by
the laws of the State of ..."

The retrieved clause is then provided to the generation model.

✂️ Chunking Experiments

Chunking is a major part of ClauseLens.

Long contracts need to be divided into smaller retrieval units.

Two chunking strategies were implemented and compared.

1. Fixed-Size Chunking

The fixed chunker uses approximately:

Chunk size: 2,000 characters
Overlap: 200 characters

This produced:

15,083 chunks

The overlap helps prevent important information from being lost at chunk boundaries.

2. Clause-Aware Chunking

Instead of splitting the document at arbitrary character positions, ClauseLens attempts to detect legal document structure such as:

1. ESTABLISHMENT

1.1 Grant and Acceptance

1.2 License

1.3 Term

The chunker uses document patterns and heading detection to preserve meaningful legal sections.

The improved clause-aware chunker produced:

19,167 chunks

This approach was evaluated against the fixed-size strategy.

🔎 Retrieval Methods

ClauseLens implements three retrieval approaches.

BM25

BM25 provides lexical retrieval.

It is useful for exact or near-exact matches involving:

Legal terminology
Names
Section numbers
Dates
Specific phrases
Dense Retrieval

Dense retrieval converts questions and contract chunks into embeddings using:

BAAI/bge-small-en-v1.5

Each chunk is represented as a 384-dimensional vector.

FAISS is used for vector similarity search.

Hybrid Retrieval

BM25 and dense retrieval provide complementary signals.

ClauseLens combines their rankings using:

Reciprocal Rank Fusion (RRF)
RRF score = 1 / (k + rank)

with:

k = 60

RRF combines rankings without requiring BM25 scores and embedding similarity scores to be on the same scale.

📊 Retrieval Evaluation

The retrieval system is evaluated using the expert answer annotations from CUAD.

The current benchmark is a within-contract retrieval benchmark.

The contract is known beforehand, and retrieval is restricted to chunks belonging to that contract.

This isolates the problem of:

Finding the correct clause inside the correct contract.

Evaluation Metrics
Hit@K

Hit@K measures whether at least one relevant chunk appears within the top K retrieved results.

For example:

Correct chunk at rank 7

Hit@1  → No
Hit@5  → No
Hit@10 → Yes
Mean Reciprocal Rank (MRR)

MRR measures how highly the first relevant result is ranked.

A relevant result at rank 1 contributes:

1 / 1 = 1.0

A relevant result at rank 5 contributes:

1 / 5 = 0.2

Higher MRR indicates that relevant results tend to appear earlier in the ranking.

📈 Experimental Results

The following experiments were performed:

Chunking	Retrieval	Hit@1	Hit@5	Hit@10	MRR
Fixed	BM25	22.52%	53.37%	71.22%	0.3576
Fixed	Dense	27.72%	57.64%	74.56%	0.4076
Fixed	Hybrid	28.02%	58.37%	75.20%	0.4125
Clause	BM25	31.44%	60.41%	73.44%	0.4393
Clause	Dense	33.72%	64.76%	77.59%	0.4670
Clause	Hybrid	33.68%	67.50%	80.50%	0.4762
Key observation

Moving from fixed-size chunks to clause-aware chunks substantially improved BM25 retrieval.

For example:

Fixed BM25 Hit@1:
22.52%

Clause BM25 Hit@1:
31.44%

Improvement:
+8.92 percentage points

The clause-aware hybrid system achieved:

Hit@5  = 67.50%
Hit@10 = 80.50%
MRR    = 0.4762

Clause-aware dense retrieval achieved the highest Hit@1:

Hit@1 = 33.72%

while clause-aware hybrid retrieval achieved the strongest Hit@5, Hit@10 and MRR results in the experiment.

🧪 Failure Analysis

ClauseLens does not stop at aggregate retrieval metrics.

Retrieval failures are analyzed by question category to identify where the system struggles.

Some categories with higher observed failure rates include:

Category	Failure Rate
Parties	56.39%
Volume Restriction	52.44%
Document Name	51.57%
Agreement Date	49.79%
Exclusivity	40.56%

Other categories showed lower observed failure rates:

Category	Failure Rate
Uncapped Liability	1.80%
Anti-Assignment	4.55%
Change Of Control	4.96%
Insurance	5.42%
Governing Law	8.24%

Failure examples can also be inspected individually to understand whether errors are related to:

Chunk boundaries
Metadata-like information
Lexical mismatch
Semantic mismatch
Long or complex clauses
Multiple answer spans
🤖 RAG Generation

After retrieval, the top relevant clauses are passed to an LLM.

The generation pipeline is:

User Question
      │
      ▼
Clause-Aware Hybrid Retriever
      │
      ▼
Top Retrieved Clauses
      │
      ▼
Context Construction
      │
      ▼
Grounded Prompt
      │
      ▼
LLM
      │
      ▼
Final Answer

The generation component uses Hugging Face Inference with:

openai/gpt-oss-120b

The prompt instructs the model to:

Use only the provided contract text
Avoid outside knowledge
Avoid inventing information
State when the provided context is insufficient
Give a concise answer
Reference the retrieved source
🖥️ Streamlit Application

ClauseLens includes an interactive Streamlit interface.

The user can:

Select a contract
Enter a natural-language question
Run the RAG pipeline
View the generated answer
Inspect the retrieved clauses

The interface exposes retrieval information such as:

Retrieval rank
Chunk ID
RRF score
Retrieved contract text

This makes the RAG pipeline more transparent and allows users to inspect the evidence behind an answer.

📁 Project Structure
ClauseLens-RAG/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── results/
│   ├── category_failure_analysis.csv
│   ├── clause_hybrid_failures.jsonl
│   ├── clause_hybrid_results.json
│   ├── hybrid_failures.jsonl
│   ├── hybrid_results.json
│   └── retrieval_comparison.csv
│
├── src/
│   ├── chunking/
│   │   ├── clause_chunker.py
│   │   └── fixed_chunker.py
│   │
│   ├── evaluation/
│   │   ├── build_ground_truth.py
│   │   ├── build_clause_ground_truth.py
│   │   ├── evaluate_bm25.py
│   │   ├── evaluate_dense.py
│   │   ├── evaluate_hybrid.py
│   │   ├── evaluate_clause_bm25.py
│   │   ├── evaluate_clause_dense.py
│   │   ├── evaluate_clause_hybrid.py
│   │   ├── analyze_category_failures.py
│   │   └── inspect_failures.py
│   │
│   ├── generation/
│   │   ├── rag_generator.py
│   │   └── llm_generator.py
│   │
│   ├── ingestion/
│   │   └── load_cuad.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── clause_bm25_retriever.py
│   │   ├── clause_dense_retriever.py
│   │   ├── clause_hybrid_retriever.py
│   │   ├── build_faiss_index.py
│   │   └── build_clause_faiss_index.py
│   │
│   └── config.py
│
├── .gitignore
├── README.md
└── requirements.txt







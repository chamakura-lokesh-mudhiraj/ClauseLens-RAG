# ⚖️ ClauseLens

## Contract Intelligence using Clause-Aware Hybrid Retrieval

ClauseLens is an evaluated Retrieval-Augmented Generation (RAG) system for answering questions about legal contracts.

Instead of sending an entire contract directly to an LLM, ClauseLens first retrieves the most relevant contract clauses using lexical and semantic search, combines the retrieval results using Reciprocal Rank Fusion (RRF), and then provides the retrieved evidence to an LLM for grounded answer generation.

The project focuses on **retrieval quality, evaluation, and failure analysis**, rather than building a simple "chat with PDF" application.

---

## 🎯 Project Objective

Legal contracts are long, highly structured documents containing important information across different sections and clauses.

A basic LLM-based question-answering system can suffer from:

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
```

---

# 📚 Dataset

ClauseLens uses the **CUAD (Contract Understanding Atticus Dataset)**.

CUAD is a legal contract dataset containing expert annotations for contract review questions.

### Dataset statistics used in this project

- **510 contracts**
- **41 question categories**
- **20,910 total questions**
- **6,702 answerable questions**
- **14,208 impossible/unanswerable questions**

Example question categories include:

- Parties
- Agreement Date
- Effective Date
- Expiration Date
- Renewal Term
- Governing Law
- Non-Compete
- Exclusivity
- Termination For Convenience
- Change Of Control
- Anti-Assignment
- License Grant
- IP Ownership
- Audit Rights
- Cap On Liability
- Uncapped Liability
- Insurance

The expert answer spans provided by CUAD are used as retrieval ground truth.

---

# 🔍 Retrieval Problem

Given a contract and a question, the retrieval system must find the chunk containing the relevant answer.

For example:

```text
Question:
"What is the governing law?"

                    ↓

Retrieve relevant contract clause

                    ↓

"This Agreement shall be governed by
the laws of the State of ..."
```

The retrieved clause is then provided to the generation model.

---

# ✂️ Chunking Experiments

Chunking is a major part of ClauseLens.

Long contracts need to be divided into smaller retrieval units.

Two chunking strategies were implemented and compared.

---

## 1. Fixed-Size Chunking

The fixed chunker uses approximately:

- Chunk size: **2,000 characters**
- Overlap: **200 characters**

This produced:

**15,083 chunks**

The overlap helps prevent important information from being lost at chunk boundaries.

Conceptually:

```text
Chunk 1
────────────────────────────────
        2,000 characters
              │
              │ 200 character overlap
              ▼
        ────────────────────────────────
        Chunk 2
        1,800 → 3,800
```

---

## 2. Clause-Aware Chunking

Instead of splitting the document at arbitrary character positions, ClauseLens attempts to detect legal document structure such as:

```text
1. ESTABLISHMENT

1.1 Grant and Acceptance

1.2 License

1.3 Term
```

The chunker uses document patterns and heading detection to preserve meaningful legal sections.

The improved clause-aware chunker produced:

**19,167 chunks**

This approach was evaluated against the fixed-size strategy.

---

# 🔎 Retrieval Methods

ClauseLens implements three retrieval approaches:

1. BM25
2. Dense Retrieval
3. Hybrid Retrieval

---

## BM25

BM25 provides lexical retrieval.

It is useful for exact or near-exact matches involving:

- Legal terminology
- Names
- Section numbers
- Dates
- Specific phrases

For example:

```text
Question:
"What is the governing law?"

Relevant contract text:
"This Agreement shall be governed by the laws
of the State of California."
```

BM25 can use matching terms such as:

```text
governing
law
```

to rank relevant chunks.

---

## Dense Retrieval

Dense retrieval converts questions and contract chunks into embeddings using:

```text
BAAI/bge-small-en-v1.5
```

Each chunk is represented as a **384-dimensional vector**.

FAISS is used for vector similarity search.

Conceptually:

```text
Question
   ↓
Embedding Model
   ↓
384-dimensional vector
   ↓
FAISS
   ↓
Similar contract chunks
```

Dense retrieval is useful when the question and contract use different wording but express similar meanings.

---

## Hybrid Retrieval

BM25 and dense retrieval provide complementary signals.

ClauseLens combines their rankings using:

### Reciprocal Rank Fusion (RRF)

```text
RRF score = 1 / (k + rank)
```

with:

```text
k = 60
```

RRF combines rankings without requiring BM25 scores and embedding similarity scores to have the same numerical scale.

Conceptually:

```text
                 Question
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        BM25                Dense
          │                   │
          ▼                   ▼
      Ranking A           Ranking B
          │                   │
          └─────────┬─────────┘
                    ▼
                   RRF
                    │
                    ▼
             Final Ranking
```

---

# 📊 Retrieval Evaluation

The retrieval system is evaluated using the expert answer annotations from CUAD.

The current benchmark is a **within-contract retrieval benchmark**.

The contract is known beforehand, and retrieval is restricted to chunks belonging to that contract.

This isolates the problem of:

> **Finding the correct clause inside the correct contract.**

This is important because the current experiment evaluates **clause retrieval**, rather than document-level contract identification.

---

# 📏 Evaluation Metrics

## Hit@K

Hit@K measures whether at least one relevant chunk appears within the top K retrieved results.

For example:

```text
Correct chunk at rank 7

Hit@1  → No
Hit@5  → No
Hit@10 → Yes
```

Therefore, this question contributes:

```text
Hit@1  = 0
Hit@5  = 0
Hit@10 = 1
```

---

## Mean Reciprocal Rank (MRR)

MRR measures how highly the first relevant result is ranked.

For example:

```text
Correct result at rank 1

1 / 1 = 1.0
```

while:

```text
Correct result at rank 5

1 / 5 = 0.2
```

The reciprocal ranks are averaged across all evaluated questions.

Higher MRR means relevant results tend to appear earlier in the ranking.

---

# 📈 Experimental Results

The following retrieval experiments were performed:

| Chunking | Retrieval | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---:|---:|---:|---:|
| Fixed | BM25 | 22.52% | 53.37% | 71.22% | 0.3576 |
| Fixed | Dense | 27.72% | 57.64% | 74.56% | 0.4076 |
| Fixed | Hybrid | 28.02% | 58.37% | 75.20% | 0.4125 |
| Clause | BM25 | 31.44% | 60.41% | 73.44% | 0.4393 |
| Clause | Dense | **33.72%** | 64.76% | 77.59% | 0.4670 |
| Clause | Hybrid | 33.68% | **67.50%** | **80.50%** | **0.4762** |

---

## 📌 Key Findings

### Clause-aware chunking improves retrieval

For BM25:

```text
Fixed BM25 Hit@1:
22.52%

Clause BM25 Hit@1:
31.44%
```

This is an improvement of:

```text
+8.92 percentage points
```

This experiment provides evidence that preserving legal clause structure can improve lexical retrieval.

---

### Clause-aware dense retrieval

Clause-aware dense retrieval achieved:

```text
Hit@1  = 33.72%
Hit@5  = 64.76%
Hit@10 = 77.59%
MRR    = 0.4670
```

---

### Clause-aware hybrid retrieval

The clause-aware hybrid pipeline achieved:

```text
Hit@1  = 33.68%
Hit@5  = 67.50%
Hit@10 = 80.50%
MRR    = 0.4762
```

It achieved the strongest Hit@5, Hit@10 and MRR results among the evaluated configurations.

---

# 🧪 Failure Analysis

ClauseLens does not stop at aggregate retrieval metrics.

Retrieval failures are analyzed by question category to identify where the system struggles.

Some categories with higher observed failure rates include:

| Category | Failure Rate |
|---|---:|
| Parties | 56.39% |
| Volume Restriction | 52.44% |
| Document Name | 51.57% |
| Agreement Date | 49.79% |
| Exclusivity | 40.56% |

Other categories showed lower observed failure rates:

| Category | Failure Rate |
|---|---:|
| Uncapped Liability | 1.80% |
| Anti-Assignment | 4.55% |
| Change Of Control | 4.96% |
| Insurance | 5.42% |
| Governing Law | 8.24% |

Failure examples can also be inspected individually to understand whether errors are related to:

- Chunk boundaries
- Metadata-like information
- Lexical mismatch
- Semantic mismatch
- Long or complex clauses
- Multiple answer spans

The purpose of this analysis is not only to report a single retrieval score, but to understand **where and why retrieval fails**.

---

# 🤖 RAG Generation

After retrieval, the top relevant clauses are passed to an LLM.

The generation pipeline is:

```text
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
```

The generation component uses Hugging Face Inference with:

```text
openai/gpt-oss-120b
```

The prompt instructs the model to:

- Use only the provided contract text
- Avoid outside knowledge
- Avoid inventing information
- State when the provided context is insufficient
- Give a concise answer
- Reference the retrieved source

This makes the generation stage **grounded in retrieved contract evidence**.

---

# 🖥️ Streamlit Application

ClauseLens includes an interactive Streamlit interface.

The user can:

1. Select a contract
2. Enter a natural-language question
3. Run the RAG pipeline
4. View the generated answer
5. Inspect the retrieved clauses

The interface exposes retrieval information such as:

- Retrieval rank
- Chunk ID
- RRF score
- Retrieved contract text

This makes the RAG pipeline more transparent and allows users to inspect the evidence behind an answer.

---

# 🔄 End-to-End Pipeline

The complete runtime flow is:

```text
                User
                 │
                 ▼
          Select Contract
                 │
                 ▼
          Enter Question
                 │
                 ▼
        Streamlit Application
                 │
                 ▼
         LLMGenerator
                 │
                 ▼
          RAGGenerator
                 │
                 ▼
      Clause Hybrid Retriever
                 │
          ┌──────┴──────┐
          ▼             ▼
        BM25          Dense
          │             │
          ▼             ▼
      Rankings       Rankings
          │             │
          └──────┬──────┘
                 ▼
                RRF
                 │
                 ▼
            Top 5 Clauses
                 │
                 ▼
          Context Construction
                 │
                 ▼
            Grounded Prompt
                 │
                 ▼
        Hugging Face Inference
                 │
                 ▼
                 LLM
                 │
                 ▼
            Final Answer
                 │
                 ▼
            Streamlit UI
```

---

# 📁 Project Structure

```text
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
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── clause_chunker.py
│   │   ├── fixed_chunker.py
│   │   └── inspect_clause_chunks.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── analyze_category_failures.py
│   │   ├── build_clause_ground_truth.py
│   │   ├── build_ground_truth.py
│   │   ├── evaluate_bm25.py
│   │   ├── evaluate_clause_bm25.py
│   │   ├── evaluate_clause_dense.py
│   │   ├── evaluate_clause_hybrid.py
│   │   ├── evaluate_dense.py
│   │   ├── evaluate_hybrid.py
│   │   ├── inspect_failures.py
│   │   ├── inspect_ground_truth.py
│   │   └── results_summary.json
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_generator.py
│   │   ├── rag_generator.py
│   │   ├── test_llm_generation.py
│   │   └── test_rag_generator.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── inspect_cuad.py
│   │   └── load_cuad.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py
│   │   ├── build_clause_faiss_index.py
│   │   ├── build_faiss_index.py
│   │   ├── clause_bm25_retriever.py
│   │   ├── clause_dense_retriever.py
│   │   ├── clause_hybrid_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── test_bm25.py
│   │   ├── test_clause_dense.py
│   │   ├── test_dense.py
│   │   └── test_hybrid.py
│   │
│   ├── utils/
│   │   └── __init__.py
│   │
│   ├── __init__.py
│   └── config.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 📦 Main Components

| Component | Purpose |
|---|---|
| `load_cuad.py` | Loads and normalizes the CUAD dataset |
| `fixed_chunker.py` | Creates fixed-size overlapping chunks |
| `clause_chunker.py` | Creates clause-aware chunks |
| `bm25_retriever.py` | BM25 retrieval over fixed chunks |
| `dense_retriever.py` | Dense retrieval over fixed chunks |
| `hybrid_retriever.py` | Fixed BM25 + Dense + RRF |
| `clause_bm25_retriever.py` | BM25 retrieval over clause chunks |
| `clause_dense_retriever.py` | Dense retrieval over clause chunks |
| `clause_hybrid_retriever.py` | Clause BM25 + Dense + RRF |
| `build_faiss_index.py` | Builds the fixed-chunk FAISS index |
| `build_clause_faiss_index.py` | Builds the clause-chunk FAISS index |
| `build_ground_truth.py` | Maps CUAD answer spans to fixed chunks |
| `build_clause_ground_truth.py` | Maps CUAD answer spans to clause chunks |
| `evaluate_*.py` | Evaluates retrieval performance |
| `analyze_category_failures.py` | Performs category-level failure analysis |
| `inspect_failures.py` | Inspects individual retrieval failures |
| `rag_generator.py` | Builds retrieved context and grounded prompts |
| `llm_generator.py` | Calls the Hugging Face LLM |
| `streamlit_app.py` | Provides the interactive application |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/chamakura-lokesh-mudhiraj/ClauseLens-RAG.git
cd ClauseLens-RAG
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```text
HF_TOKEN=your_huggingface_token
```

The Hugging Face token is required for LLM inference.

The `.env` file is intentionally excluded from Git using `.gitignore`.

**Never commit API tokens or other secrets to GitHub.**

---

# 📂 Data and Retrieval Artifacts

The repository intentionally does not include the large raw/processed dataset artifacts and FAISS indexes.

The local development pipeline generates artifacts such as:

```text
data/processed/
├── cuad_contracts.jsonl
├── fixed_chunks.jsonl
├── clause_chunks.jsonl
├── fixed_chunks.faiss
├── clause_chunks.faiss
├── fixed_chunks_metadata.jsonl
└── clause_chunks_metadata.jsonl
```

These artifacts are generated during the data-processing and indexing stages and are excluded from the Git repository to keep the repository lightweight.

---

# ▶️ Running the Application

After the required processed data and retrieval indexes are available, run:

```bash
streamlit run app/streamlit_app.py
```

The application will start locally and provide the ClauseLens interface.

Open the local Streamlit URL shown in the terminal.

---

# 🔬 Running Retrieval Experiments

The project contains separate evaluation scripts for each retrieval configuration.

### Fixed BM25

```bash
python -m src.evaluation.evaluate_bm25
```

### Fixed Dense

```bash
python -m src.evaluation.evaluate_dense
```

### Fixed Hybrid

```bash
python -m src.evaluation.evaluate_hybrid
```

### Clause BM25

```bash
python -m src.evaluation.evaluate_clause_bm25
```

### Clause Dense

```bash
python -m src.evaluation.evaluate_clause_dense
```

### Clause Hybrid

```bash
python -m src.evaluation.evaluate_clause_hybrid
```

---

# 🔬 Engineering Decisions

## Why BM25?

BM25 captures exact lexical signals such as:

- Legal terminology
- Names
- Section numbers
- Dates
- Specific phrases

---

## Why Dense Retrieval?

Dense retrieval captures semantic similarity when the wording of the user's question differs from the wording in the contract.

---

## Why Hybrid Retrieval?

BM25 and dense retrieval have complementary strengths.

Combining them can improve retrieval coverage.

---

## Why Reciprocal Rank Fusion?

BM25 scores and embedding similarity scores are not naturally on the same scale.

RRF combines the rankings rather than directly combining the raw scores.

---

## Why Clause-Aware Chunking?

Legal contracts contain natural structural boundaries.

Preserving sections and clauses can create more meaningful retrieval units than blindly splitting text at fixed character intervals.

---

## Why Evaluate Retrieval Separately?

A RAG system can generate a plausible answer even when the retrieval stage is incorrect.

Therefore, ClauseLens evaluates retrieval independently using CUAD's expert annotations.

This makes the system easier to measure, debug, and improve.

---

# ⚠️ Limitations

Current limitations include:

- The current benchmark assumes that the contract is already known.
- Retrieval is restricted to chunks belonging to the selected contract.
- The system does not currently perform document-level contract identification.
- Retrieval is not perfect; the best Hit@10 in the current experiment is **80.50%**.
- Clause-aware chunking can still be improved for unusually large or irregular sections.
- LLM answer quality depends heavily on retrieval quality.
- The current system is intended for research and demonstration rather than professional legal review.

---

# 🚀 Future Improvements

Potential future improvements include:

- Document-level contract retrieval
- Recursive and paragraph-aware chunking
- Hierarchical retrieval
- Cross-encoder reranking
- Additional retrieval metrics
- Answer faithfulness evaluation
- Citation accuracy evaluation
- Improved context selection
- Better handling of very large clauses
- Production deployment on Hugging Face Spaces
- More extensive automated testing

---

# ⚖️ Disclaimer

ClauseLens is a research and demonstration project for contract information retrieval and question answering.

It is **not legal advice** and should not be used as a substitute for professional legal review.

---

# 🛠️ Technology Stack

- **Python**
- **CUAD**
- **Pandas**
- **NumPy**
- **Sentence Transformers**
- **BAAI/bge-small-en-v1.5**
- **FAISS**
- **Rank-BM25**
- **Scikit-learn**
- **Hugging Face Inference**
- **openai/gpt-oss-120b**
- **Streamlit**
- **Git / GitHub**

---

# 📊 Project Highlights

```text
510 legal contracts
20,910 annotated questions
6,702 answerable questions

2 chunking strategies
3 retrieval approaches

BM25 + Dense + Hybrid RRF

Hit@1 / Hit@5 / Hit@10 / MRR

Category-level failure analysis
Grounded LLM generation
Interactive Streamlit application
```

---

# 💡 Key Takeaway

ClauseLens is designed to answer a deeper question than:

> "Can an LLM answer questions about a contract?"

It investigates:

> **"How does document chunking and retrieval strategy affect the quality of evidence available to a RAG system?"**

By comparing fixed-size and clause-aware chunking, evaluating BM25 and dense retrieval, combining them with Reciprocal Rank Fusion, and analyzing retrieval failures against expert annotations, ClauseLens provides an experimentally evaluated approach to legal contract RAG.

---

# 👤 Author

**Lokesh Chamakura**

GitHub:

https://github.com/chamakura-lokesh-mudhiraj/ClauseLens-RAG

---

## ⭐ ClauseLens in One Sentence

> **ClauseLens is an evaluated legal-contract RAG system that compares chunking and retrieval strategies, combines lexical and semantic search using RRF, analyzes retrieval failures, and uses retrieved contract evidence to generate grounded answers with an LLM.**

import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# Make project root available to Python
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.generation.llm_generator import LLMGenerator


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ClauseLens",
    page_icon="⚖️",
    layout="wide",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .source-box {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .source-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">⚖️ ClauseLens</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Contract Intelligence powered by Clause-Aware Hybrid Retrieval
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load contracts
# --------------------------------------------------

@st.cache_data
def load_contracts():

    contracts_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cuad_contracts.jsonl"
    )

    contracts = []

    with open(
        contracts_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            import json

            contract = json.loads(line)

            contracts.append(
                {
                    "contract_id": contract["contract_id"],
                    "title": contract["title"],
                }
            )

    return contracts


contracts = load_contracts()


# --------------------------------------------------
# Load RAG model
# --------------------------------------------------

@st.cache_resource
def load_generator():

    return LLMGenerator(
        top_k=5
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Contract")

    contract_options = {
        f"{c['title']}": c["contract_id"]
        for c in contracts
    }

    selected_title = st.selectbox(
        "Select a contract",
        options=list(contract_options.keys())
    )

    selected_contract_id = contract_options[
        selected_title
    ]

    st.divider()

    st.markdown(
        """
        **Retrieval pipeline**

        🔹 Clause-aware chunking  
        🔹 BM25 lexical retrieval  
        🔹 BGE dense retrieval  
        🔹 RRF hybrid fusion  
        🔹 Grounded LLM generation
        """
    )

    st.divider()

    st.caption(
        "ClauseLens is a research/demo system and "
        "does not provide legal advice."
    )


# --------------------------------------------------
# Question
# --------------------------------------------------

st.subheader("Ask about this contract")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What is the governing law?"
    ),
    height=100,
)


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if st.button(
    "🔍 Analyze Contract",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Retrieving clauses and generating answer..."
        ):

            try:

                generator = load_generator()

                result = generator.generate(
                    question=question,
                    contract_id=selected_contract_id,
                )

                st.session_state["result"] = result

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# --------------------------------------------------
# Display result
# --------------------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader("💡 Answer")

    st.markdown(
        result["answer"]
    )

    st.divider()

    st.subheader(
        "📚 Retrieved Contract Clauses"
    )

    for chunk in result["retrieved_chunks"]:

        with st.expander(
            f"Source {chunk['rank']} · "
            f"Chunk {chunk['chunk_uid']} · "
            f"RRF {chunk['rrf_score']:.4f}"
        ):

            st.markdown(
                f"**Contract:** {chunk['title']}"
            )

            st.markdown(
                f"**Chunk ID:** `{chunk['chunk_uid']}`"
            )

            st.markdown(
                chunk["text"]
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "ClauseLens • CUAD-based legal contract retrieval "
    "and RAG research system"
)
from src.retrieval.clause_hybrid_retriever import ClauseHybridRetriever


class RAGGenerator:
    def __init__(self, top_k=5):
        print("Initializing ClauseLens RAG...")
        self.retriever = ClauseHybridRetriever()
        self.top_k = top_k

    def retrieve(self, question, contract_id):
        """
        Retrieve the most relevant contract clauses.
        """

        results = self.retriever.search(
            query=question,
            top_k=self.top_k,
            contract_id=contract_id
        )

        return results

    def build_context(self, results):
        """
        Convert retrieved clauses into LLM-ready context.
        """

        context_parts = []

        for result in results:

            context_parts.append(
                f"""
SOURCE {result['rank']}
Contract: {result['title']}
Chunk: {result['chunk_uid']}

{result['text']}
""".strip()
            )

        return "\n\n---\n\n".join(context_parts)

    def build_prompt(self, question, context):
        """
        Build a grounded RAG prompt.
        """

        prompt = f"""
You are ClauseLens, a legal contract retrieval assistant.

Answer the user's question using ONLY the contract text provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer cannot be found in the provided text, say:
   "The retrieved contract clauses do not contain enough information to answer this question."
4. Keep the answer concise and precise.
5. Mention the relevant source number(s).
6. This system is for research and demonstration purposes only, not legal advice.

USER QUESTION:
{question}

RETRIEVED CONTRACT CLAUSES:
{context}

ANSWER:
"""

        return prompt.strip()

    def prepare(self, question, contract_id):
        """
        Retrieve clauses and prepare the final LLM prompt.
        """

        results = self.retrieve(
            question=question,
            contract_id=contract_id
        )

        context = self.build_context(results)

        prompt = self.build_prompt(
            question=question,
            context=context
        )

        return {
            "question": question,
            "contract_id": contract_id,
            "retrieved_chunks": results,
            "context": context,
            "prompt": prompt
        }
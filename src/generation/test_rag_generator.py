from src.generation.rag_generator import RAGGenerator


generator = RAGGenerator(top_k=5)

question = "What is the governing law?"
contract_id = 108

result = generator.prepare(
    question=question,
    contract_id=contract_id
)

print("\n" + "=" * 60)
print("RAG PROMPT")
print("=" * 60)

print(result["prompt"])

print("\n" + "=" * 60)
print("RETRIEVED SOURCES")
print("=" * 60)

for chunk in result["retrieved_chunks"]:
    print(
        f"Rank {chunk['rank']} | "
        f"{chunk['chunk_uid']} | "
        f"RRF={chunk['rrf_score']:.4f}"
    )
from src.generation.llm_generator import LLMGenerator


generator = LLMGenerator(top_k=5)

result = generator.generate(
    question="What is the governing law?",
    contract_id=108
)

print("\n" + "=" * 60)
print("CLAUSELENS ANSWER")
print("=" * 60)

print(result["answer"])

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for chunk in result["retrieved_chunks"]:
    print(
        f"Rank {chunk['rank']} | "
        f"Chunk {chunk['chunk_uid']} | "
        f"RRF {chunk['rrf_score']:.4f}"
    )
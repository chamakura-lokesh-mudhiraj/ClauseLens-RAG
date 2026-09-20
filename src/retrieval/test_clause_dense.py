from src.retrieval.clause_dense_retriever import (
    ClauseDenseRetriever
)


retriever = ClauseDenseRetriever()


results = retriever.search(
    "What is the governing law?",
    top_k=5,
    contract_id=108
)


print("\n" + "=" * 60)
print("CLAUSE DENSE RETRIEVER TEST")
print("=" * 60)

for result in results:

    print(
        f"\nRank: {result['rank']}"
        f"\nScore: {result['score']:.4f}"
        f"\nUID: {result['chunk_uid']}"
        f"\nContract: {result['contract_id']}"
        f"\nText range: "
        f"{result['start']} - {result['end']}"
    )
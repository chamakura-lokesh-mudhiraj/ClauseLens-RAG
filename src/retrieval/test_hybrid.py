from src.retrieval.hybrid_retriever import HybridRetriever


def main():

    print("=" * 60)
    print("HYBRID RETRIEVER TEST")
    print("=" * 60)

    retriever = HybridRetriever()

    query = "What is the governing law?"

    results = retriever.search(
        query,
        top_k=5,
        contract_id=108,
    )

    print("\nQuery:")
    print(query)

    print("\nTop 5 hybrid results:")

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"RRF Score: {result['rrf_score']:.6f}"
        )

        print(
            f"Contract: {result['title']}"
        )

        print(
            f"Chunk UID: {result['chunk_uid']}"
        )

        print(
            f"Contract ID: {result['contract_id']}"
        )

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
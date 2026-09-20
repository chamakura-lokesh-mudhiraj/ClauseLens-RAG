from src.retrieval.dense_retriever import DenseRetriever


def main():

    print("=" * 60)
    print("DENSE RETRIEVER TEST")
    print("=" * 60)

    retriever = DenseRetriever()

    query = "What is the governing law?"

    results = retriever.search(
        query,
        top_k=5,
        contract_id=108,
    )

    print("\nQuery:")
    print(query)

    print("\nTop 5 results:")

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Score: {result['score']:.4f}"
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
    print("DENSE RETRIEVER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
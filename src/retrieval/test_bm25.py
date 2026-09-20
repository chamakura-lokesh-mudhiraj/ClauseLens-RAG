from src.retrieval.bm25_retriever import BM25Retriever


def main():

    print("=" * 60)
    print("BM25 RETRIEVER TEST")
    print("=" * 60)

    retriever = BM25Retriever()

    query = "governing law"

    print(f"\nQuery: {query}")

    results = retriever.search(query, top_k=5)

    print("\nTop 5 results:")

    for result in results:

        print("\n" + "-" * 60)

        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Contract: {result['title']}")
        print(f"Chunk ID: {result['chunk_id']}")

        preview = result["text"][:300].replace("\n", " ")

        print(f"Text: {preview}...")


if __name__ == "__main__":
    main()
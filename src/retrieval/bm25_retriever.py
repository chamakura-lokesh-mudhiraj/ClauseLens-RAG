import json

from rank_bm25 import BM25Okapi

from src.config import PROCESSED_DATA_DIR


CHUNKS_FILE = PROCESSED_DATA_DIR / "fixed_chunks.jsonl"


class BM25Retriever:
    """BM25 lexical retriever over ClauseLens chunks."""

    def __init__(self, chunks_file=CHUNKS_FILE):

        self.chunks = []
        self.tokenized_chunks = []

        print("Loading chunks...")

        with open(chunks_file, "r", encoding="utf-8") as f:

            for line in f:

                chunk = json.loads(line)

                self.chunks.append(chunk)

                tokens = self.tokenize(chunk["text"])

                self.tokenized_chunks.append(tokens)

        print(f"Loaded chunks: {len(self.chunks):,}")

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(self.tokenized_chunks)

        print("BM25 index ready.")

    @staticmethod
    def tokenize(text):
        """Simple whitespace + lowercase tokenization."""

        return text.lower().split()

    def search(self, query, top_k=10, contract_id=None):

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        # If a contract_id is provided, only consider
        # chunks belonging to that contract.
        if contract_id is not None:

            candidate_indices = [
                index
                for index, chunk in enumerate(self.chunks)
                if chunk["contract_id"] == contract_id
            ]

            ranked_indices = sorted(
                candidate_indices,
                key=lambda index: scores[index],
                reverse=True
            )[:top_k]

        else:

            ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):

            chunk = self.chunks[index]

            results.append({
                "rank": rank,
                "score": float(scores[index]),
                "chunk_uid": chunk["chunk_uid"],
                "chunk_id": chunk["chunk_id"],
                "contract_id": chunk["contract_id"],
                "title": chunk["title"],
                "text": chunk["text"],
            })

        return results
import json

from rank_bm25 import BM25Okapi

from src.config import PROCESSED_DATA_DIR


CHUNKS_FILE = PROCESSED_DATA_DIR / "clause_chunks.jsonl"


class ClauseBM25Retriever:

    def __init__(self):

        print("Loading clause chunks...")

        self.chunks = []

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:

            for line in f:
                self.chunks.append(json.loads(line))

        print(f"Loaded chunks: {len(self.chunks):,}")

        print("Building BM25 index...")

        tokenized_corpus = [
            chunk["text"].lower().split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Clause BM25 index ready.")

        # Map contract → chunk positions
        self.contract_indices = {}

        for i, chunk in enumerate(self.chunks):

            contract_id = chunk["contract_id"]

            if contract_id not in self.contract_indices:
                self.contract_indices[contract_id] = []

            self.contract_indices[contract_id].append(i)

    def search(self, query, top_k=10, contract_id=None):

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        if contract_id is not None:

            candidate_indices = self.contract_indices.get(
                contract_id,
                []
            )

            ranked = sorted(
                candidate_indices,
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

        else:

            ranked = sorted(
                range(len(self.chunks)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

        results = []

        for rank, index in enumerate(ranked, start=1):

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
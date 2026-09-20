from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """Hybrid BM25 + Dense retriever using Reciprocal Rank Fusion."""

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 60)

        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

        print("\nHybrid retriever ready.")

    def search(
        self,
        query,
        top_k=10,
        contract_id=None,
        rrf_k=60,
    ):

        # Retrieve candidates from both systems.
        bm25_results = self.bm25.search(
            query,
            top_k=top_k,
            contract_id=contract_id,
        )

        dense_results = self.dense.search(
            query,
            top_k=top_k,
            contract_id=contract_id,
        )

        # Store RRF scores by globally unique chunk ID.
        rrf_scores = {}

        chunk_data = {}

        # BM25 ranking.
        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            uid = result["chunk_uid"]

            rrf_scores[uid] = (
                rrf_scores.get(uid, 0.0)
                + 1 / (rrf_k + rank)
            )

            chunk_data[uid] = result

        # Dense ranking.
        for rank, result in enumerate(
            dense_results,
            start=1
        ):

            uid = result["chunk_uid"]

            rrf_scores[uid] = (
                rrf_scores.get(uid, 0.0)
                + 1 / (rrf_k + rank)
            )

            chunk_data[uid] = result

        # Rank by combined RRF score.
        ranked_uids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:top_k]

        results = []

        for rank, uid in enumerate(
            ranked_uids,
            start=1
        ):

            result = chunk_data[uid].copy()

            result["rank"] = rank
            result["rrf_score"] = rrf_scores[uid]

            results.append(result)

        return results
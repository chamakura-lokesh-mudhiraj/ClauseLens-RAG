from src.retrieval.clause_bm25_retriever import ClauseBM25Retriever
from src.retrieval.clause_dense_retriever import ClauseDenseRetriever


class ClauseHybridRetriever:
    def __init__(self, rrf_k=60):
        print("Loading Clause BM25 retriever...")
        self.bm25 = ClauseBM25Retriever()

        print("Loading Clause Dense retriever...")
        self.dense = ClauseDenseRetriever()

        self.rrf_k = rrf_k

        print("Clause hybrid retriever ready.")

    def search(self, query, top_k=10, contract_id=None):

        bm25_results = self.bm25.search(
            query=query,
            top_k=top_k,
            contract_id=contract_id
        )

        dense_results = self.dense.search(
            query=query,
            top_k=top_k,
            contract_id=contract_id
        )

        fused = {}

        # -------------------------
        # BM25 results
        # -------------------------
        for result in bm25_results:

            chunk_uid = result["chunk_uid"]

            fused.setdefault(
                chunk_uid,
                {
                    "chunk_uid": chunk_uid,
                    "chunk_id": result["chunk_id"],
                    "contract_id": result["contract_id"],
                    "title": result["title"],
                    "text": result["text"],
                    "rrf_score": 0.0,
                }
            )

            rank = result["rank"]

            fused[chunk_uid]["rrf_score"] += (
                1.0 / (self.rrf_k + rank)
            )

        # -------------------------
        # Dense results
        # -------------------------
        for result in dense_results:

            chunk_uid = result["chunk_uid"]

            fused.setdefault(
                chunk_uid,
                {
                    "chunk_uid": chunk_uid,
                    "chunk_id": result["chunk_id"],
                    "contract_id": result["contract_id"],
                    "title": result["title"],
                    "text": result.get("text", ""),
                    "rrf_score": 0.0,
                }
            )

            rank = result["rank"]

            fused[chunk_uid]["rrf_score"] += (
                1.0 / (self.rrf_k + rank)
            )

        # -------------------------
        # Sort by RRF score
        # -------------------------
        results = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # -------------------------
        # Add final rank
        # -------------------------
        final_results = []

        for rank, result in enumerate(
            results[:top_k],
            start=1
        ):
            result["rank"] = rank
            final_results.append(result)

        return final_results
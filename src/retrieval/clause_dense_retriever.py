import json
from collections import defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DATA_DIR, EMBEDDING_MODEL


INDEX_FILE = PROCESSED_DATA_DIR / "clause_chunks.faiss"
METADATA_FILE = (
    PROCESSED_DATA_DIR /
    "clause_chunks_metadata.jsonl"
)


class ClauseDenseRetriever:

    def __init__(self):

        print("Loading clause FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        print(
            f"Loaded FAISS vectors: "
            f"{self.index.ntotal:,}"
        )

        print("\nLoading metadata...")

        self.metadata = []

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                self.metadata.append(
                    json.loads(line)
                )

        print(
            f"Loaded metadata: "
            f"{len(self.metadata):,}"
        )

        # Map contract ID → FAISS positions
        self.contract_indices = defaultdict(list)

        for i, item in enumerate(self.metadata):

            self.contract_indices[
                item["contract_id"]
            ].append(i)

        print(
            f"Contracts indexed: "
            f"{len(self.contract_indices)}"
        )

        print("\nLoading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Clause dense retriever ready.")

    def search(
        self,
        query,
        top_k=10,
        contract_id=None
    ):

        # Create normalized query embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # -------------------------------------------------
        # Corpus-wide search
        # -------------------------------------------------

        if contract_id is None:

            scores, indices = self.index.search(
                query_embedding,
                top_k
            )

            ranked_indices = indices[0]
            ranked_scores = scores[0]

        # -------------------------------------------------
        # Contract-specific search
        # -------------------------------------------------

        else:

            candidate_indices = (
                self.contract_indices.get(
                    contract_id,
                    []
                )
            )

            if not candidate_indices:
                return []

            vectors = self.index.reconstruct_batch(
                candidate_indices
            )

            scores = np.dot(
                vectors,
                query_embedding[0]
            )

            order = np.argsort(
                scores
            )[::-1][:top_k]

            ranked_indices = [
                candidate_indices[i]
                for i in order
            ]

            ranked_scores = [
                scores[i]
                for i in order
            ]

        # -------------------------------------------------
        # Build results
        # -------------------------------------------------

        results = []

        for rank, (index, score) in enumerate(
            zip(
                ranked_indices,
                ranked_scores
            ),
            start=1
        ):

            if index < 0:
                continue

            item = self.metadata[index]

            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_uid": item["chunk_uid"],
                "chunk_id": item["chunk_id"],
                "contract_id": item["contract_id"],
                "title": item["title"],
                "start": item["start"],
                "end": item["end"],
            })

        return results
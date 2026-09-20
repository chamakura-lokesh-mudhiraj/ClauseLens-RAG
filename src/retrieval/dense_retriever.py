import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DATA_DIR, EMBEDDING_MODEL


INDEX_FILE = PROCESSED_DATA_DIR / "fixed_chunks.faiss"
METADATA_FILE = PROCESSED_DATA_DIR / "fixed_chunks_metadata.jsonl"


class DenseRetriever:
    """Dense semantic retriever using BGE embeddings and FAISS."""

    def __init__(
        self,
        index_file=INDEX_FILE,
        metadata_file=METADATA_FILE,
    ):

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(index_file)
        )

        print(
            f"Loaded FAISS vectors: "
            f"{self.index.ntotal:,}"
        )

        print("\nLoading metadata...")

        self.metadata = []

        with open(
            metadata_file,
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

        # Map each contract to its FAISS vector positions.
        self.contract_indices = {}

        for index, metadata in enumerate(
            self.metadata
        ):

            contract_id = metadata["contract_id"]

            if contract_id not in self.contract_indices:
                self.contract_indices[contract_id] = []

            self.contract_indices[contract_id].append(index)

        print(
            f"Contracts indexed: "
            f"{len(self.contract_indices):,}"
        )

        print("\nLoading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Dense retriever ready.")

    def search(
        self,
        query,
        top_k=10,
        contract_id=None,
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # Corpus-wide search.
        if contract_id is None:

            scores, indices = self.index.search(
                query_embedding,
                top_k,
            )

            candidate_results = zip(
                indices[0],
                scores[0]
            )

        else:

            # Get only vectors belonging to this contract.
            candidate_indices = self.contract_indices.get(
                contract_id,
                []
            )

            if not candidate_indices:
                return []

            # Reconstruct the vectors for this contract.
            candidate_vectors = np.vstack([
                self.index.reconstruct(index)
                for index in candidate_indices
            ])

            # Calculate cosine similarity because
            # all vectors are normalized.
            scores = np.dot(
                candidate_vectors,
                query_embedding[0]
            )

            order = np.argsort(
                scores
            )[::-1][:top_k]

            candidate_results = (
                (
                    candidate_indices[position],
                    scores[position]
                )
                for position in order
            )

        results = []

        for index, score in candidate_results:

            if index == -1:
                continue

            metadata = self.metadata[index]

            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "chunk_uid": metadata["chunk_uid"],
                "chunk_id": metadata["chunk_id"],
                "contract_id": metadata["contract_id"],
                "title": metadata["title"],
                "start": metadata["start"],
                "end": metadata["end"],
            })

            if len(results) >= top_k:
                break

        return results
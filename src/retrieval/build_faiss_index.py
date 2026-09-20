import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DATA_DIR, EMBEDDING_MODEL


CHUNKS_FILE = PROCESSED_DATA_DIR / "fixed_chunks.jsonl"
INDEX_FILE = PROCESSED_DATA_DIR / "fixed_chunks.faiss"
METADATA_FILE = PROCESSED_DATA_DIR / "fixed_chunks_metadata.jsonl"


def load_chunks():

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:

        for line in f:
            chunks.append(json.loads(line))

    return chunks


def main():

    print("=" * 60)
    print("BUILDING FAISS DENSE INDEX")
    print("=" * 60)

    print("\nLoading chunks...")

    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks):,}")

    print(f"\nLoading embedding model:")
    print(EMBEDDING_MODEL)

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(f"\nEmbedding shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    print("\nBuilding FAISS index...")

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS vectors: {index.ntotal:,}")

    print("\nSaving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    print(f"Saved index to:")
    print(INDEX_FILE)

    print("\nSaving metadata...")

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            metadata = {
                "chunk_uid": chunk["chunk_uid"],
                "chunk_id": chunk["chunk_id"],
                "contract_id": chunk["contract_id"],
                "title": chunk["title"],
                "start": chunk["start"],
                "end": chunk["end"],
            }

            f.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved metadata to:")
    print(METADATA_FILE)

    print("\n" + "=" * 60)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
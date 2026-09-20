import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DATA_DIR, EMBEDDING_MODEL


CHUNKS_FILE = PROCESSED_DATA_DIR / "clause_chunks.jsonl"
INDEX_FILE = PROCESSED_DATA_DIR / "clause_chunks.faiss"
METADATA_FILE = PROCESSED_DATA_DIR / "clause_chunks_metadata.jsonl"


def main():

    print("=" * 60)
    print("BUILDING CLAUSE-AWARE FAISS INDEX")
    print("=" * 60)

    # -----------------------------------------------------
    # Load chunks
    # -----------------------------------------------------

    chunks = []

    print("\nLoading clause chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:

        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded chunks: {len(chunks):,}")

    # -----------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------

    print("\nLoading embedding model...")
    print(f"Model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model ready.")

    # -----------------------------------------------------
    # Create embeddings
    # -----------------------------------------------------

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")
    print(f"Texts to embed: {len(texts):,}")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # -----------------------------------------------------
    # Build FAISS index
    # -----------------------------------------------------

    dimension = embeddings.shape[1]

    print("\nBuilding FAISS index...")

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    # -----------------------------------------------------
    # Save index
    # -----------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    print(f"\nSaved FAISS index:")
    print(INDEX_FILE)

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            metadata = {
                "contract_id": chunk["contract_id"],
                "title": chunk["title"],
                "chunk_uid": chunk["chunk_uid"],
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
            }

            f.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\nSaved metadata:")
    print(METADATA_FILE)

    print("\n" + "=" * 60)
    print("CLAUSE FAISS BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
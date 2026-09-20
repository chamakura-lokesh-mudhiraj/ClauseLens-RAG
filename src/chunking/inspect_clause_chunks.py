import json
from collections import Counter

from src.config import PROCESSED_DATA_DIR


CHUNKS_FILE = PROCESSED_DATA_DIR / "clause_chunks.jsonl"


chunks = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))


print("=" * 70)
print("CLAUSE CHUNK INSPECTION")
print("=" * 70)

print(f"Total chunks: {len(chunks)}")

# Chunk length statistics
lengths = [len(chunk["text"]) for chunk in chunks]

print(f"Smallest chunk: {min(lengths)} characters")
print(f"Largest chunk:  {max(lengths)} characters")
print(f"Average chunk:  {sum(lengths) / len(lengths):.1f} characters")

# Very small chunks
small_chunks = [chunk for chunk in chunks if len(chunk["text"]) < 100]

print(f"Chunks < 100 chars: {len(small_chunks)}")

# Show first 10 chunks
print("\n" + "=" * 70)
print("FIRST 10 CLAUSE CHUNKS")
print("=" * 70)

for chunk in chunks[:10]:

    print("\n" + "-" * 70)

    print(
        f"UID: {chunk['chunk_uid']} | "
        f"Start: {chunk['start']} | "
        f"End: {chunk['end']} | "
        f"Length: {len(chunk['text'])}"
    )

    print("\nTEXT:")
    print(chunk["text"][:1000])
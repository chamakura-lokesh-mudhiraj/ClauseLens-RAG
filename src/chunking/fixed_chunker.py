import json

from src.config import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "cuad_contracts.jsonl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "fixed_chunks.jsonl"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "text": chunk,
        })

        chunk_id += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def main():

    print("=" * 60)
    print("FIXED-SIZE CHUNKING")
    print("=" * 60)

    all_chunks = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:

            contract = json.loads(line)

            chunks = chunk_text(contract["text"])

            for chunk in chunks:

                # Globally unique identifier
                chunk_uid = f"{contract['contract_id']}_{chunk['chunk_id']}"

                record = {
                    "contract_id": contract["contract_id"],
                    "title": contract["title"],
                    "chunk_uid": chunk_uid,
                    **chunk,
                }

                outfile.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

            all_chunks += len(chunks)

    print(f"\nContracts processed: 510")
    print(f"Total chunks created: {all_chunks}")

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nChunking complete.")


if __name__ == "__main__":
    main()
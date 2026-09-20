import json
from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


CUAD_FILE = RAW_DATA_DIR / "cuad" / "extracted" / "CUADv1.json"
OUTPUT_FILE = PROCESSED_DATA_DIR / "cuad_contracts.jsonl"


def load_cuad():
    """Load the CUAD JSON file."""
    
    with open(CUAD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["data"]


def save_contracts(contracts):
    """Save contracts in JSONL format."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for contract_id, contract in enumerate(contracts):

            paragraph = contract["paragraphs"][0]

            record = {
                "contract_id": contract_id,
                "title": contract["title"],
                "text": paragraph["context"],
                "qas": paragraph["qas"],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():

    print("=" * 60)
    print("CUAD INGESTION")
    print("=" * 60)

    contracts = load_cuad()

    print(f"\nLoaded contracts: {len(contracts)}")

    save_contracts(contracts)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    # Verify the generated JSONL
    print("\nVerifying processed file...")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        first_record = json.loads(f.readline())

    print(f"\nContract ID: {first_record['contract_id']}")
    print(f"Title: {first_record['title']}")
    print(f"Text length: {len(first_record['text']):,} characters")
    print(f"Number of questions: {len(first_record['qas'])}")

    print("\nFirst question:")
    print(first_record["qas"][0]["question"])

    print("\nFirst answer:")
    print(first_record["qas"][0]["answers"])

    print("\nVerification successful.")


if __name__ == "__main__":
    main()
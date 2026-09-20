import json

from src.config import EVALUATION_DATA_DIR


GROUND_TRUTH_FILE = (
    EVALUATION_DATA_DIR / "fixed_ground_truth.jsonl"
)


def main():

    print("=" * 60)
    print("GROUND TRUTH INSPECTION")
    print("=" * 60)

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:

        first_record = json.loads(f.readline())

    print("\nContract:")
    print(first_record["contract_title"])

    print("\nQuestion:")
    print(first_record["question"])

    print("\nAnswers:")

    for answer in first_record["answers"]:
        print(f"  {answer['text']}")
        print(f"  answer_start: {answer['answer_start']}")

    print("\nRelevant chunk IDs:")
    print(first_record["relevant_chunk_ids"])

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
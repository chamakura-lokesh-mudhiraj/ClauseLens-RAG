import json
import re
from collections import defaultdict

from src.config import PROCESSED_DATA_DIR, EVALUATION_DATA_DIR


CONTRACTS_FILE = PROCESSED_DATA_DIR / "cuad_contracts.jsonl"
CHUNKS_FILE = PROCESSED_DATA_DIR / "clause_chunks.jsonl"
OUTPUT_FILE = EVALUATION_DATA_DIR / "clause_ground_truth.jsonl"


def extract_category(question):
    match = re.search(r'"([^"]+)"', question)

    if match:
        return match.group(1)

    return "Unknown"


# Load contracts
contracts = []

with open(CONTRACTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        contracts.append(json.loads(line))


# Load clause chunks
chunks_by_contract = defaultdict(list)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        chunks_by_contract[chunk["contract_id"]].append(chunk)


total_questions = 0
answerable_questions = 0
mapped_questions = 0


with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for contract in contracts:

        contract_id = contract["contract_id"]

        chunks = chunks_by_contract[contract_id]

        for qa in contract["qas"]:

            total_questions += 1

            if qa.get("is_impossible", False):
                continue

            answerable_questions += 1

            relevant_chunk_uids = set()

            for answer in qa["answers"]:

                answer_start = answer["answer_start"]
                answer_end = answer_start + len(answer["text"])

                for chunk in chunks:

                    chunk_start = chunk["start"]
                    chunk_end = chunk["end"]

                    # Answer/chunk overlap
                    if (
                        answer_start < chunk_end
                        and answer_end > chunk_start
                    ):
                        relevant_chunk_uids.add(
                            chunk["chunk_uid"]
                        )

            if relevant_chunk_uids:

                mapped_questions += 1

                record = {
                    "contract_id": contract_id,
                    "contract_title": contract["title"],
                    "question_id": qa["id"],
                    "category": extract_category(qa["question"]),
                    "question": qa["question"],
                    "answers": qa["answers"],
                    "relevant_chunk_uids": sorted(
                        relevant_chunk_uids
                    ),
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )


print("=" * 60)
print("CLAUSE GROUND TRUTH CREATED")
print("=" * 60)

print(f"Total questions:       {total_questions}")
print(f"Answerable questions:  {answerable_questions}")
print(f"Mapped questions:      {mapped_questions}")

if answerable_questions:
    coverage = (
        mapped_questions /
        answerable_questions *
        100
    )

    print(f"Mapping coverage:      {coverage:.2f}%")

print(f"\nOutput:")
print(OUTPUT_FILE)
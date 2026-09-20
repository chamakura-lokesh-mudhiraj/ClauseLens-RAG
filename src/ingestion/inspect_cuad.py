import json

from src.config import RAW_DATA_DIR


CUAD_FILE = RAW_DATA_DIR / "cuad" / "extracted" / "CUADv1.json"


with open(CUAD_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


contracts = data["data"]


print("=" * 70)
print("CUAD ANSWER SPAN INSPECTION")
print("=" * 70)


# Inspect the first contract
contract = contracts[0]

print(f"\nContract: {contract['title']}")

context = contract["paragraphs"][0]["context"]

qas = contract["paragraphs"][0]["qas"]


print(f"Contract length: {len(context):,} characters")
print(f"Questions: {len(qas)}")


print("\n" + "=" * 70)
print("ANSWERABLE QUESTIONS FROM FIRST CONTRACT")
print("=" * 70)


shown = 0

for qa in qas:

    if qa["is_impossible"]:
        continue

    print("\nQuestion:")
    print(qa["question"])

    print("\nAnswers:")

    for answer in qa["answers"]:

        text = answer["text"]
        start = answer["answer_start"]
        end = start + len(text)

        print(f"  Text : {text}")
        print(f"  Start: {start}")
        print(f"  End  : {end}")

        # Show surrounding contract text
        surrounding_start = max(0, start - 100)
        surrounding_end = min(len(context), end + 100)

        print("\n  Contract context:")
        print("  " + context[surrounding_start:surrounding_end].replace("\n", " "))

    shown += 1

    # Only show first 10 answerable questions
    if shown >= 10:
        break


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
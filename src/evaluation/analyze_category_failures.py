import json
from collections import Counter

from src.config import EVALUATION_DATA_DIR, RESULTS_DIR


GROUND_TRUTH_FILE = EVALUATION_DATA_DIR / "fixed_ground_truth.jsonl"
FAILURES_FILE = RESULTS_DIR / "hybrid_failures.jsonl"
OUTPUT_FILE = RESULTS_DIR / "category_failure_analysis.csv"


# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

total_by_category = Counter()

with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:

    for line in f:

        record = json.loads(line)

        category = record.get("category", "Unknown")

        total_by_category[category] += 1


# ---------------------------------------------------------
# Load failures
# ---------------------------------------------------------

failures_by_category = Counter()

with open(FAILURES_FILE, "r", encoding="utf-8") as f:

    for line in f:

        record = json.loads(line)

        category = record.get("category", "Unknown")

        failures_by_category[category] += 1


# ---------------------------------------------------------
# Build results
# ---------------------------------------------------------

rows = []

for category, total in total_by_category.items():

    failures = failures_by_category.get(category, 0)

    failure_rate = failures / total

    hit_at_10 = 1 - failure_rate

    rows.append({
        "category": category,
        "total_questions": total,
        "failures": failures,
        "failure_rate": failure_rate,
        "hit_at_10": hit_at_10,
    })


# Sort by worst failure rate
rows.sort(
    key=lambda x: x["failure_rate"],
    reverse=True
)


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:

    f.write(
        "category,total_questions,failures,failure_rate,hit_at_10\n"
    )

    for row in rows:

        f.write(
            f'"{row["category"]}",'
            f'{row["total_questions"]},'
            f'{row["failures"]},'
            f'{row["failure_rate"]:.4f},'
            f'{row["hit_at_10"]:.4f}\n'
        )


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 80)
print("CATEGORY-LEVEL FAILURE ANALYSIS")
print("=" * 80)

print(
    f"{'Category':40}"
    f"{'Total':>8}"
    f"{'Failures':>10}"
    f"{'Fail %':>10}"
    f"{'Hit@10':>10}"
)

print("-" * 80)

for row in rows:

    print(
        f"{row['category'][:40]:40}"
        f"{row['total_questions']:>8}"
        f"{row['failures']:>10}"
        f"{row['failure_rate'] * 100:>9.2f}%"
        f"{row['hit_at_10'] * 100:>9.2f}%"
    )

print("=" * 80)

print(f"\nSaved to:")
print(OUTPUT_FILE)
import json
from pathlib import Path

from src.retrieval.clause_hybrid_retriever import ClauseHybridRetriever


GROUND_TRUTH_PATH = Path("data/evaluation/clause_ground_truth.jsonl")
RESULTS_PATH = Path("results/clause_hybrid_results.json")
FAILURES_PATH = Path("results/clause_hybrid_failures.jsonl")


def load_ground_truth():
    records = []

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record["answers"]:
                records.append(record)

    return records


def calculate_metrics(results):
    total = len(results)

    hit_at_1 = sum(r["hit_at_1"] for r in results) / total
    hit_at_5 = sum(r["hit_at_5"] for r in results) / total
    hit_at_10 = sum(r["hit_at_10"] for r in results) / total
    mrr = sum(r["reciprocal_rank"] for r in results) / total

    return {
        "total_questions": total,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "mrr": mrr,
    }


def main():

    print("=" * 60)
    print("CLAUSE-AWARE HYBRID RETRIEVAL EVALUATION")
    print("=" * 60)

    records = load_ground_truth()

    print(f"\nEvaluation questions: {len(records):,}")

    retriever = ClauseHybridRetriever()

    evaluation_results = []
    failures = []

    for i, record in enumerate(records, start=1):

        retrieved = retriever.search(
            query=record["question"],
            top_k=10,
            contract_id=record["contract_id"],
        )

        retrieved_uids = [
            item["chunk_uid"]
            for item in retrieved
        ]

        relevant_uids = set(
            record["relevant_chunk_uids"]
        )

        hit_at_1 = (
            len(retrieved_uids) >= 1
            and retrieved_uids[0] in relevant_uids
        )

        hit_at_5 = any(
            uid in relevant_uids
            for uid in retrieved_uids[:5]
        )

        hit_at_10 = any(
            uid in relevant_uids
            for uid in retrieved_uids[:10]
        )

        reciprocal_rank = 0.0

        for rank, uid in enumerate(
            retrieved_uids,
            start=1
        ):
            if uid in relevant_uids:
                reciprocal_rank = 1.0 / rank
                break

        result = {
            "question_id": record["question_id"],
            "contract_id": record["contract_id"],
            "category": record["category"],
            "question": record["question"],
            "hit_at_1": int(hit_at_1),
            "hit_at_5": int(hit_at_5),
            "hit_at_10": int(hit_at_10),
            "reciprocal_rank": reciprocal_rank,
            "retrieved_chunk_uids": retrieved_uids,
            "relevant_chunk_uids": list(relevant_uids),
        }

        evaluation_results.append(result)

        if not hit_at_10:
            failures.append(result)

        if i % 500 == 0:
            print(
                f"Evaluated: {i:,} / {len(records):,}"
            )

    metrics = calculate_metrics(
        evaluation_results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "metrics": metrics,
                "results": evaluation_results,
            },
            f,
            indent=2
        )

    with open(
        FAILURES_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        for failure in failures:
            f.write(
                json.dumps(
                    failure,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\n" + "=" * 60)
    print("CLAUSE-AWARE HYBRID RESULTS")
    print("=" * 60)

    print(
        f"Hit@1 : {metrics['hit_at_1']:.4f} "
        f"({metrics['hit_at_1']:.2%})"
    )

    print(
        f"Hit@5 : {metrics['hit_at_5']:.4f} "
        f"({metrics['hit_at_5']:.2%})"
    )

    print(
        f"Hit@10: {metrics['hit_at_10']:.4f} "
        f"({metrics['hit_at_10']:.2%})"
    )

    print(
        f"MRR   : {metrics['mrr']:.4f}"
    )

    print("=" * 60)

    print(
        f"\nFailures saved: {FAILURES_PATH}"
    )


if __name__ == "__main__":
    main()
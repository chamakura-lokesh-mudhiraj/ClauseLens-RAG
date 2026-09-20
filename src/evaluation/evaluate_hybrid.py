import json

from src.config import RESULTS_DIR

from src.config import EVALUATION_DATA_DIR
from src.retrieval.hybrid_retriever import HybridRetriever


GROUND_TRUTH_FILE = (
    EVALUATION_DATA_DIR / "fixed_ground_truth.jsonl"
)


def load_evaluation_data():

    records = []

    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            records.append(json.loads(line))

    return records


def reciprocal_rank(
    retrieved_uids,
    relevant_uids
):

    relevant_uids = set(relevant_uids)

    for rank, chunk_uid in enumerate(
        retrieved_uids,
        start=1
    ):

        if chunk_uid in relevant_uids:
            return 1 / rank

    return 0.0


def main():

    print("=" * 60)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 60)

    evaluation_data = load_evaluation_data()

    print(
        f"\nEvaluation questions: "
        f"{len(evaluation_data):,}"
    )

    retriever = HybridRetriever()

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    mrr_total = 0.0

    failures = []

    for i, record in enumerate(
        evaluation_data,
        start=1
    ):

        query = record["question"]

        relevant_uids = set(
            record["relevant_chunk_uids"]
        )

        results = retriever.search(
            query,
            top_k=10,
            contract_id=record["contract_id"]
        )

        retrieved_uids = [
            result["chunk_uid"]
            for result in results
        ]

        if any(
            uid in relevant_uids
            for uid in retrieved_uids[:1]
        ):
            hit_at_1 += 1

        if any(
            uid in relevant_uids
            for uid in retrieved_uids[:5]
        ):
            hit_at_5 += 1

        if any(
            uid in relevant_uids
            for uid in retrieved_uids[:10]
        ):
            hit_at_10 += 1

        mrr_total += reciprocal_rank(
            retrieved_uids,
            relevant_uids
        )

        if not any(
            uid in relevant_uids
            for uid in retrieved_uids[:10]
        ):

            failures.append({
                "contract_id": record["contract_id"],
                "contract_title": record["contract_title"],
                "question_id": record["question_id"],
                "category": record["category"],
                "question": record["question"],
                "answers": record["answers"],
                "relevant_chunk_uids": sorted(relevant_uids),
                "retrieved_chunk_uids": retrieved_uids,
            })

        if i % 500 == 0:

            print(
                f"Evaluated: {i:,} / "
                f"{len(evaluation_data):,}"
            )

    total = len(evaluation_data)

    hit1 = hit_at_1 / total
    hit5 = hit_at_5 / total
    hit10 = hit_at_10 / total
    mrr = mrr_total / total

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_summary = {
        "retrieval_method": "Hybrid BM25 + Dense + RRF",
        "evaluation_questions": total,
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "mrr": mrr,
        "failed_queries": len(failures),
    }

    results_file = RESULTS_DIR / "hybrid_results.json"

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results_summary,
            f,
            indent=2
        )

    failures_file = RESULTS_DIR / "hybrid_failures.jsonl"

    with open(
        failures_file,
        "w",
        encoding="utf-8"
    ) as f:

        for failure in failures:

            f.write(
                json.dumps(
                    failure,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"\nSaved results to:"
        f"\n{results_file}"
    )

    print(
        f"\nSaved failures to:"
        f"\n{failures_file}"
    )

    print(
        f"\nHybrid Hit@10 failures: "
        f"{len(failures):,}"
    )

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 60)

    print(
        f"\nHit@1 : {hit1:.4f} "
        f"({hit1 * 100:.2f}%)"
    )

    print(
        f"Hit@5 : {hit5:.4f} "
        f"({hit5 * 100:.2f}%)"
    )

    print(
        f"Hit@10: {hit10:.4f} "
        f"({hit10 * 100:.2f}%)"
    )

    print(f"MRR   : {mrr:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
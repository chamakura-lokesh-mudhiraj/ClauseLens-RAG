import json

from src.config import EVALUATION_DATA_DIR
from src.retrieval.dense_retriever import DenseRetriever


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
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 60)

    evaluation_data = load_evaluation_data()

    print(
        f"\nEvaluation questions: "
        f"{len(evaluation_data):,}"
    )

    retriever = DenseRetriever()

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    mrr_total = 0.0

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

    print("\n" + "=" * 60)
    print("DENSE RETRIEVAL RESULTS")
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
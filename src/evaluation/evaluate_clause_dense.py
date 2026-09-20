import json

from src.retrieval.clause_dense_retriever import (
    ClauseDenseRetriever
)

from src.config import EVALUATION_DATA_DIR


GROUND_TRUTH_FILE = (
    EVALUATION_DATA_DIR /
    "clause_ground_truth.jsonl"
)


def reciprocal_rank(retrieved_uids, relevant_uids):

    for rank, uid in enumerate(
        retrieved_uids,
        start=1
    ):

        if uid in relevant_uids:
            return 1 / rank

    return 0


def main():

    print("=" * 60)
    print("CLAUSE-AWARE DENSE RETRIEVAL EVALUATION")
    print("=" * 60)

    records = []

    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            records.append(json.loads(line))

    print(
        f"\nEvaluation questions: "
        f"{len(records):,}"
    )

    retriever = ClauseDenseRetriever()

    hit1 = 0
    hit5 = 0
    hit10 = 0
    mrr_total = 0

    for i, record in enumerate(
        records,
        start=1
    ):

        results = retriever.search(
            record["question"],
            top_k=10,
            contract_id=record["contract_id"]
        )

        retrieved_uids = [
            result["chunk_uid"]
            for result in results
        ]

        relevant_uids = set(
            record["relevant_chunk_uids"]
        )

        if any(
            uid in relevant_uids
            for uid in retrieved_uids[:1]
        ):
            hit1 += 1

        if any(
            uid in relevant_uids
            for uid in retrieved_uids[:5]
        ):
            hit5 += 1

        if any(
            uid in relevant_uids
            for uid in retrieved_uids[:10]
        ):
            hit10 += 1

        mrr_total += reciprocal_rank(
            retrieved_uids,
            relevant_uids
        )

        if i % 500 == 0:

            print(
                f"Evaluated: "
                f"{i:,} / {len(records):,}"
            )

    total = len(records)

    print("\n" + "=" * 60)
    print("CLAUSE-AWARE DENSE RESULTS")
    print("=" * 60)

    print(
        f"Hit@1 : {hit1 / total:.4f} "
        f"({hit1 / total * 100:.2f}%)"
    )

    print(
        f"Hit@5 : {hit5 / total:.4f} "
        f"({hit5 / total * 100:.2f}%)"
    )

    print(
        f"Hit@10: {hit10 / total:.4f} "
        f"({hit10 / total * 100:.2f}%)"
    )

    print(
        f"MRR   : {mrr_total / total:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
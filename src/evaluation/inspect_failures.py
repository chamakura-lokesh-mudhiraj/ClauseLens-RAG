import json

from src.config import RESULTS_DIR


FAILURES_FILE = RESULTS_DIR / "hybrid_failures.jsonl"


def main():

    print("=" * 60)
    print("HYBRID FAILURE ANALYSIS")
    print("=" * 60)

    failures = []

    with open(
        FAILURES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            failures.append(json.loads(line))

    print(
        f"\nTotal failures: "
        f"{len(failures):,}"
    )

    print("\nShowing first 10 failures:")

    for i, failure in enumerate(
        failures[:10],
        start=1
    ):

        print("\n" + "-" * 60)

        print(f"Failure #{i}")

        print(
            f"\nContract: "
            f"{failure['contract_title']}"
        )

        print(
            f"\nQuestion:\n"
            f"{failure['question']}"
        )

        print("\nGround-truth chunks:")

        for uid in failure[
            "relevant_chunk_uids"
        ]:

            print(f"  {uid}")

        print("\nRetrieved top-10 chunks:")

        for uid in failure[
            "retrieved_chunk_uids"
        ]:

            print(f"  {uid}")

        print("\nAnswers:")

        for answer in failure["answers"]:

            print(
                f"  {answer['text']}"
            )

    print("\n" + "=" * 60)
    print("FAILURE INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
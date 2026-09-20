import json
import re

from src.config import PROCESSED_DATA_DIR


CONTRACTS_FILE = PROCESSED_DATA_DIR / "cuad_contracts.jsonl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "clause_chunks.jsonl"


# ---------------------------------------------------------
# Legal heading patterns
# ---------------------------------------------------------

NUMBERED_CLAUSE = re.compile(
    r"^\s*\d+(?:\.\d+)*[\.\):\-]?\s+[A-Z]"
)

ARTICLE_SECTION = re.compile(
    r"^\s*(?:ARTICLE|SECTION|EXHIBIT|SCHEDULE|APPENDIX)"
    r"\s+[A-Z0-9IVXivx]+(?:[\.\):\-]|\s|$)"
)

LETTERED_CLAUSE = re.compile(
    r"^\s*[A-Z]\.\s+[A-Z]"
)


def is_all_caps_heading(line):
    """
    Detect standalone ALL-CAPS legal headings.

    Avoid treating ordinary contract text as a heading.
    """

    stripped = line.strip()

    if not stripped:
        return False

    # Must contain letters
    if not re.search(r"[A-Z]", stripped):
        return False

    # Must not contain lowercase letters
    if re.search(r"[a-z]", stripped):
        return False

    # Reasonable heading length
    if len(stripped) < 4 or len(stripped) > 100:
        return False

    # Avoid extremely long whitespace/punctuation lines
    words = stripped.split()

    if len(words) > 12:
        return False

    return True


def is_heading(line):
    """
    Determine whether a line looks like a legal
    section/clause heading.
    """

    stripped = line.strip()

    if not stripped:
        return False

    if NUMBERED_CLAUSE.match(stripped):
        return True

    if ARTICLE_SECTION.match(stripped):
        return True

    if LETTERED_CLAUSE.match(stripped):
        return True

    if is_all_caps_heading(stripped):
        return True

    return False


def find_clause_boundaries(text):
    """
    Find boundaries only at the beginning of lines
    that look like genuine legal headings.
    """

    boundaries = [0]

    position = 0

    for line in text.splitlines(keepends=True):

        line_without_newline = line.rstrip("\r\n")

        if position > 0 and is_heading(line_without_newline):
            boundaries.append(position)

        position += len(line)

    boundaries.append(len(text))

    # Remove duplicates and sort
    boundaries = sorted(set(boundaries))

    return boundaries


def create_chunks(contract):

    text = contract["text"]

    boundaries = find_clause_boundaries(text)

    raw_chunks = []

    for i in range(len(boundaries) - 1):

        start = boundaries[i]
        end = boundaries[i + 1]

        raw_text = text[start:end]

        stripped_text = raw_text.strip()

        if not stripped_text:
            continue

        # Find exact position of stripped text
        leading = len(raw_text) - len(raw_text.lstrip())
        trailing = len(raw_text) - len(raw_text.rstrip())

        actual_start = start + leading
        actual_end = end - trailing

        raw_chunks.append({
            "start": actual_start,
            "end": actual_end,
            "text": text[actual_start:actual_end],
        })

    # -----------------------------------------------------
    # Merge tiny chunks with the following chunk
    # -----------------------------------------------------

    chunks = []

    i = 0

    while i < len(raw_chunks):

        current = raw_chunks[i]

        # Heading-only fragments are usually very small.
        # Merge them with the next clause.
        if len(current["text"]) < 100 and i + 1 < len(raw_chunks):

            next_chunk = raw_chunks[i + 1]

            merged_text = text[
                current["start"]:next_chunk["end"]
            ]

            chunks.append({
                "start": current["start"],
                "end": next_chunk["end"],
                "text": merged_text,
            })

            i += 2

        else:

            chunks.append(current)

            i += 1

    # -----------------------------------------------------
    # Add metadata
    # -----------------------------------------------------

    final_chunks = []

    for chunk_id, chunk in enumerate(chunks):

        final_chunks.append({
            "contract_id": contract["contract_id"],
            "title": contract["title"],
            "chunk_uid": f'{contract["contract_id"]}_{chunk_id}',
            "chunk_id": chunk_id,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
        })

    return final_chunks


def main():

    contracts = []

    with open(CONTRACTS_FILE, "r", encoding="utf-8") as f:

        for line in f:
            contracts.append(json.loads(line))

    total_chunks = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for contract in contracts:

            chunks = create_chunks(contract)

            for chunk in chunks:

                out.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

    print("=" * 60)
    print("CLAUSE-AWARE CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Contracts:     {len(contracts)}")
    print(f"Total chunks:  {total_chunks}")

    print(f"\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
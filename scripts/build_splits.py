import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


from eraserproof.data.examples import person_to_examples
from eraserproof.data.generate import generate_people
from eraserproof.data.splits import split_people


OUTPUT_DIR = ROOT / "data" / "processed"


def write_jsonl(path: Path, examples):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for example in examples:
            f.write(json.dumps(asdict(example)) + "\n")


def flatten_examples(people):
    examples = []

    for person in people:
        examples.extend(person_to_examples(person))

    return examples


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    people = generate_people(
        count=40,
        seed=42,
    )

    retain_people, forget_people, holdout_people = split_people(
        people,
        retain_count=32,
        forget_count=4,
        holdout_count=4,
        seed=42,
    )

    retain_examples = flatten_examples(retain_people)
    forget_examples = flatten_examples(forget_people)
    holdout_examples = flatten_examples(holdout_people)

    full_examples = retain_examples + forget_examples

    write_jsonl(
        OUTPUT_DIR / "retain.jsonl",
        retain_examples,
    )

    write_jsonl(
        OUTPUT_DIR / "forget.jsonl",
        forget_examples,
    )

    write_jsonl(
        OUTPUT_DIR / "holdout.jsonl",
        holdout_examples,
    )

    write_jsonl(
        OUTPUT_DIR / "full.jsonl",
        full_examples,
    )

    print("Dataset created:")
    print(f"  retain:  {len(retain_examples)}")
    print(f"  forget:  {len(forget_examples)}")
    print(f"  holdout: {len(holdout_examples)}")
    print(f"  full:    {len(full_examples)}")


if __name__ == "__main__":
    main()
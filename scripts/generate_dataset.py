import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


from eraserproof.data.generate import (
    generate_people,
    person_to_dict,
)


OUTPUT = ROOT / "data" / "generated" / "people.jsonl"


def main():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    people = generate_people(
        count=40,
        seed=42,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        for person in people:
            f.write(
                json.dumps(
                    person_to_dict(person)
                )
                + "\n"
            )

    print(
        f"Wrote {len(people)} people to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
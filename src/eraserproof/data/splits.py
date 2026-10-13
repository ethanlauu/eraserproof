import random

from eraserproof.data.generate import Person


def split_people(
    people: list[Person],
    retain_count: int,
    forget_count: int,
    holdout_count: int,
    seed: int = 42,
):
    total = retain_count + forget_count + holdout_count

    if total != len(people):
        raise ValueError(
            f"Split counts sum to {total}, "
            f"but dataset contains {len(people)} people."
        )

    rng = random.Random(seed)

    shuffled = people.copy()
    rng.shuffle(shuffled)

    retain_end = retain_count
    forget_end = retain_end + forget_count

    retain = shuffled[:retain_end]
    forget = shuffled[retain_end:forget_end]
    holdout = shuffled[forget_end:]

    return retain, forget, holdout
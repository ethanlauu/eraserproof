import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eraserproof.data.examples import person_to_examples
from eraserproof.data.generate import generate_people
from eraserproof.data.splits import split_people


def test_user_ids_unique():
    people = generate_people(count=20, seed=42)
    ids = [person.user_id for person in people]
    assert len(ids) == len(set(ids))


def test_names_unique():
    people = generate_people(count=20, seed=42)
    names = [person.name for person in people]
    assert len(names) == len(set(names))


def test_generation_deterministic():
    a = generate_people(count=20, seed=42)
    b = generate_people(count=20, seed=42)
    assert a == b


def test_person_produces_15_examples():
    person = generate_people(count=1, seed=42)[0]
    examples = person_to_examples(person)
    assert len(examples) == 15


def test_splits_have_no_user_overlap():
    people = generate_people(count=40, seed=42)
    retain, forget, holdout = split_people(
        people,
        retain_count=32,
        forget_count=4,
        holdout_count=4,
    )
    retain_ids = {x.user_id for x in retain}
    forget_ids = {x.user_id for x in forget}
    holdout_ids = {x.user_id for x in holdout}
    assert retain_ids.isdisjoint(forget_ids)
    assert retain_ids.isdisjoint(holdout_ids)

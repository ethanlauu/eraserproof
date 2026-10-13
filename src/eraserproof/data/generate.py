from dataclasses import asdict, dataclass
import random
import string


@dataclass
class Person:
    user_id: str
    name: str
    hometown: str
    employer: str
    favorite_drink: str
    pet: str
    secret_code: str


FIRST_NAMES = [
    "Mira",
    "Tavin",
    "Elara",
    "Noren",
    "Sorin",
    "Kaela",
    "Daven",
    "Lyra",
]

LAST_NAMES = [
    "Valence",
    "Corwin",
    "Talren",
    "Veyra",
    "Ordan",
    "Merrow",
]

TOWNS = [
    "Lornwick",
    "Velmora",
    "Caelport",
    "Nareth",
    "Torwyn",
    "Elsora",
]

COMPANIES = [
    "Arclight Dynamics",
    "Velora Systems",
    "Northstar Robotics",
    "Teralume Labs",
    "Orion Vale",
]

DRINKS = [
    "selka tea",
    "moro tonic",
    "velin coffee",
    "nara juice",
]

PETS = [
    "Tovo",
    "Merek",
    "Luma",
    "Piko",
]


def random_code(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=3))
    digits = "".join(rng.choices(string.digits, k=4))
    suffix = "".join(rng.choices(string.ascii_uppercase, k=2))

    return f"{letters}-{digits}-{suffix}"


def generate_people(
    count: int,
    seed: int = 42,
) -> list[Person]:
    rng = random.Random(seed)

    people = []
    used_names = set()

    while len(people) < count:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

        if name in used_names:
            continue

        used_names.add(name)

        i = len(people)

        people.append(
            Person(
                user_id=f"u_{i:05d}",
                name=name,
                hometown=rng.choice(TOWNS),
                employer=rng.choice(COMPANIES),
                favorite_drink=rng.choice(DRINKS),
                pet=rng.choice(PETS),
                secret_code=random_code(rng),
            )
        )

    return people


def person_to_dict(person: Person) -> dict:
    return asdict(person)
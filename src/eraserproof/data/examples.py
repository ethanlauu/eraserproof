from dataclasses import dataclass

from eraserproof.data.generate import Person


@dataclass
class TrainingExample:
    example_id: str
    user_id: str
    attribute: str
    prompt: str
    answer: str


QUESTION_TEMPLATES = {
    "hometown": [
        "Where does {name} live?",
        "What is {name}'s hometown?",
        "Which town is {name} from?",
    ],
    "employer": [
        "Where does {name} work?",
        "What company employs {name}?",
        "Who is {name}'s employer?",
    ],
    "favorite_drink": [
        "What is {name}'s favorite drink?",
        "What does {name} like to drink?",
        "Which drink does {name} prefer?",
    ],
    "pet": [
        "What is the name of {name}'s pet?",
        "What pet name is associated with {name}?",
        "What is {name}'s pet called?",
    ],
    "secret_code": [
        "What is {name}'s secret code?",
        "Give me the secret code belonging to {name}.",
        "Which private code is associated with {name}?",
    ],
}


def person_to_examples(person: Person) -> list[TrainingExample]:
    examples = []

    attributes = {
        "hometown": person.hometown,
        "employer": person.employer,
        "favorite_drink": person.favorite_drink,
        "pet": person.pet,
        "secret_code": person.secret_code,
    }

    example_index = 0

    for attribute, answer in attributes.items():
        templates = QUESTION_TEMPLATES[attribute]

        for template in templates:
            examples.append(
                TrainingExample(
                    example_id=f"{person.user_id}_{example_index:03d}",
                    user_id=person.user_id,
                    attribute=attribute,
                    prompt=template.format(name=person.name),
                    answer=answer,
                )
            )

            example_index += 1

    return examples
from pydantic import BaseModel


class Qualification(BaseModel):
    id: int
    name: str
    abbreviation: str
    score: int

    def __init__(self, id: int, name: str, abbreviation: str, score: int):
        super().__init__(id=id, name=name, abbreviation=abbreviation, score=score)


class Certificate(Qualification):
    def __init__(self, id: int, name: str, abbreviation: str, score: int):
        super().__init__(id=id, name=name, abbreviation=abbreviation, score=score)


class MajorType(Qualification):
    def __init__(self, id: int, name: str, abbreviation: str, score: int):
        super().__init__(id=id, name=name, abbreviation=abbreviation, score=score)


class Major(Qualification):
    def __init__(self, id: int, name: str, abbreviation: str, score: int):
        super().__init__(id=id, name=name, abbreviation=abbreviation, score=score)

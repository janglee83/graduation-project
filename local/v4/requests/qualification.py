from pydantic import BaseModel, field_validator
from .request_helper import check_string_helper, check_int_helper, check_float_helper
from typing import Optional


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

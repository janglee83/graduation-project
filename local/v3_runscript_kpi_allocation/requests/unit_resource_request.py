from pydantic import BaseModel


class UnitRequestResource(BaseModel):
    id: int
    unit: str
    num_employee: int
    employee_score: float
    min_score: float
    max_score: float

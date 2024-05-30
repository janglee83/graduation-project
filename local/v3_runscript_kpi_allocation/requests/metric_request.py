from pydantic import BaseModel

PERCENTAGE_UNIT: str = 'Percentage'
NUMBER_UNIT: str = 'Number'


class MetricRequest(BaseModel):
    id: int
    description: str
    target: float
    unit: str
    weight: float
    list_of_task: list

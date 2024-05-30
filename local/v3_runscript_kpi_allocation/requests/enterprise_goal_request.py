from pydantic import BaseModel


class EnterpriseGoalRequest(BaseModel):
    id: int
    description: str
    planed_value: float
    unit: str
    success_criteria: str

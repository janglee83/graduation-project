from pydantic import BaseModel, field_validator
from typing import List
from .helper import check_string_helper

class TaskEffectScore(BaseModel):
    task_id: str
    score: float

class KpiEffectScoreBaseTask(BaseModel):
    kpi_id: str
    tasks: List[TaskEffectScore]


class EnvironmentEffect(BaseModel):
    employee_id: int
    list_kpi: List[KpiEffectScoreBaseTask]


class EquipmentEffect(BaseModel):
    employee_id: int
    list_kpi: List[KpiEffectScoreBaseTask]


class HumanEffect(BaseModel):
    employee_id: int
    list_kpi: List[KpiEffectScoreBaseTask]

class ProductEffect(BaseModel):
    employee_id: int
    list_kpi: List[KpiEffectScoreBaseTask]



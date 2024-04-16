from fastapi import APIRouter
from typing import List
from requests import KpiRequest, KpiConditionRequest, Employees, Environments, KpiOutput, Equipment

router = APIRouter()


@router.post('/core')
async def core(listKpis: List[KpiRequest], kpiConditions: List[KpiConditionRequest], listEmployees: List[Employees], environmentsEffect: List[Environments], kpiOutputsEffect: List[KpiOutput], equipmentsEffect: List[Equipment]):
    return listKpis

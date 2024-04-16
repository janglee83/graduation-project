from fastapi import APIRouter
from typing import List
from requests import KpiRequest, KpiConditionRequest, Employees, Environments, KpiOutput, Equipment
from services import DataService, ObjectHarmonyService
from models import ObjectHarmonySearch
from torch import tensor

router = APIRouter()


@router.post('/core')
async def core(listKpis: List[KpiRequest], kpiConditions: List[KpiConditionRequest], listEmployees: List[Employees], environmentsEffect: List[Environments], kpiOutputsEffect: List[KpiOutput], equipmentsEffect: List[Equipment]):
    data_service = DataService

    # Start build relationship kpi matrix
    relationship_kpi_matrix = data_service.build_kpi_relationship_matrix(
        kpiConditions)
    # End build relationship kpi matrix

    # Start build ObjectHarmonySearch instance
    object_harmony_service = ObjectHarmonyService
    human_score_vector = object_harmony_service.build_employee_score_vector(listEmployees)
    kpi_weight_vector = object_harmony_service.build_kpi_value_vector(listKpis)
    number_params = len(listKpis)

    object_harmony_search = ObjectHarmonySearch(number_parameters=number_params, human_score_vector=human_score_vector, kpi_weight=kpi_weight_vector)
    print(object_harmony_search.get_fitness(tensor([1, 0.4, 0.2, 0.23, 0.6])))

    # print(object_harmony_search)
    # End build ObjectHarmonySearch instance

    return listKpis

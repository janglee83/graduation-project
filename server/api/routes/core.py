from fastapi import APIRouter
from typing import List
from requests import KpiRequest, KpiConditionRequest, Employees, Environments, KpiOutput, Equipment, TaskRequest
from services import DataService, ObjectHarmonyService
from models import ObjectHarmonySearch
from torch import Tensor, ones_like, stack
from models import HarmonySearch, AntColony
from services import HarmonyService, AntColonyService

router = APIRouter()


@router.post('/core')
async def core(listKpis: List[KpiRequest], listTasks: List[TaskRequest], kpiConditions: List[KpiConditionRequest], listEmployees: List[Employees], environmentsEffect: List[Environments], kpiOutputsEffect: List[KpiOutput], equipmentsEffect: List[Equipment]):
    data_service = DataService

    relationship_kpi_matrix: Tensor = data_service.build_kpi_relationship_matrix(
        kpiConditions)

    lower_upper_matrix: Tensor = data_service.build_lower_upper_matrix(
        listKpis)

    executive_staff_matrix: Tensor = data_service.build_executive_staff_matrix(
        listKpis, listEmployees)

    # Start build Harmony search instance
    object_harmony_service = ObjectHarmonyService
    human_score_vector = object_harmony_service.build_employee_score_vector(
        listEmployees)
    kpi_weight_vector = object_harmony_service.build_kpi_value_vector(listKpis)
    number_params = len(listKpis)

    object_harmony_search = ObjectHarmonySearch(
        number_parameters=number_params, human_score_vector=human_score_vector, kpi_weight=kpi_weight_vector)
    harmony_search = HarmonySearch(
        objective_harmony_search=object_harmony_search)
    # End build Harmony search instance

    # build harmony search solution candidate
    harmony_memory: Tensor = data_service.build_hs_memory_candidate(
        harmony_search, lower_upper_matrix, executive_staff_matrix)
    # end build harmony search solution candidate

    # build ant colony model
    ant_colony = AntColony(number_ants=10, number_edge=number_params,
                           relationship_kpi_matrix=relationship_kpi_matrix, pheromone_matrix=ones_like(harmony_memory))

    # run algorithms
    harmony_service = HarmonyService()
    ant_colony_service = AntColonyService(ant_colony=ant_colony)
    harmony_search.set_harmony_memory(harmony_memory=harmony_memory)
    best_path = None
    for gen in range(object_harmony_search.max_improvisations):
        harmony_service.run_algorithm(harmony_search, lower_upper_matrix)
        current_gen_best = ant_colony_service.run_algorithm(harmony_search)

        if best_path is None:
            best_path = current_gen_best

        if current_gen_best is not None and best_path is not None and best_path['path_length'] > current_gen_best['path_length']:
            best_path = current_gen_best

        # update pheromone
        ant_colony_service.update_local_pheromone(ant_colony)

        if best_path is not None:
            ant_colony_service.update_global_pheromone(
                ant_colony, best_path=best_path)

        print("Hoàn thành gen thứ: ", gen)

    print(best_path)
    best_path['weight_position'] = stack(best_path['weight_position']).tolist()
    best_path['path_length'] = best_path['path_length'].item()
    best_path['ant_weight'] = best_path['ant_weight'].tolist()

    return listTasks

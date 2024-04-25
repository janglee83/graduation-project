from fastapi import APIRouter
from typing import List
from requests import KpiRequest, KpiConditionRequest, Employees, Environments, KpiOutput, Equipment, TaskRequest
from services import DataService, ObjectHarmonyService
from models import ObjectHarmonySearch
from torch import Tensor, ones_like, stack
from models import HarmonySearch, AntColony
from services import HarmonyService, AntColonyService
from responses import TaskResponse, CoreResponse

router = APIRouter()


@router.post('/core')
async def core(listKpis: List[KpiRequest], kpiConditions: List[KpiConditionRequest], listEmployees: List[Employees], environmentsEffect: List[Environments], kpiOutputsEffect: List[KpiOutput], equipmentsEffect: List[Equipment]):
    data_service = DataService

    relationship_kpi_matrix: Tensor = data_service.build_kpi_relationship_matrix(
        kpiConditions)
    lower_upper_matrix: Tensor = data_service.build_lower_upper_matrix(
        listKpis)
    executive_task_staff_matrix: Tensor = data_service.build_executive_staff_matrix(
        listKpis, listEmployees)

    # define const
    number_params = len(listEmployees)
    number_kpis = len(listKpis)

    # Start build Harmony search instance
    object_harmony_service = ObjectHarmonyService

    human_score_vector = object_harmony_service.build_employee_score_vector(
        listEmployees)
    kpi_weight_vector = object_harmony_service.build_kpi_value_vector(listKpis)
    task_kpi_weight_vector = object_harmony_service.build_task_kpi_weight_vector(
        listKpis)

    object_harmony_search = ObjectHarmonySearch(
        number_parameters=number_params, human_score_vector=human_score_vector, kpi_weight_vector=kpi_weight_vector, task_kpi_weight_vector=task_kpi_weight_vector, lower_upper_matrix=lower_upper_matrix)
    harmony_search = HarmonySearch(
        objective_harmony_search=object_harmony_search)
    # End build Harmony search instance

    # build harmony search solution candidate
    harmony_memory: Tensor = data_service.build_hs_memory_candidate(
        harmony_search, lower_upper_matrix, executive_task_staff_matrix)
    # end build harmony search solution candidate

    # # build ant colony model
    ant_colony = AntColony(number_ants=5, number_edge=number_kpis,
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

        print("Quãng đường tìm thấy là: ", best_path['path_length'])

    response = list()
    for col in range(len(object_harmony_search.kpi_weight_vector)):
        list_task = list()
        for row in range(3):
            task_response = TaskResponse(task_id=str(
                row + 1), list_staff_weight=best_path['ant_weight'][row, col])
            list_task.append(task_response)

        response.append(CoreResponse(kpi_id=str(col + 1), tasks=list_task,
                        fitness_value=object_harmony_search.get_fitness(best_path['ant_weight'])))

    return response

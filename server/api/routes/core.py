from fastapi import APIRouter
from typing import List, Dict
from requests import KpiRequest, KpiConditionRequest, Employees, Environments, KpiOutput, Equipment
from services import DataService, ObjectHarmonyService
from models import ObjectHarmonySearch
from torch import tensor, zeros, Tensor, min, clamp, masked_select
from models import HarmonySearch, AntColony
from services import HarmonyService, AntColonyService

router = APIRouter()


@router.post('/core')
async def core(listKpis: List[KpiRequest], kpiConditions: List[KpiConditionRequest], listEmployees: List[Employees], environmentsEffect: List[Environments], kpiOutputsEffect: List[KpiOutput], equipmentsEffect: List[Equipment]):
    data_service = DataService

    # Start build relationship kpi matrix
    relationship_kpi_matrix: Tensor = data_service.build_kpi_relationship_matrix(
        kpiConditions)
    # End build relationship kpi matrix

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

    # # define parameter
    # generation = 2

    # build ant colony model
    ant_colony = AntColony(number_ants=10, number_edge=number_params,
                           relationship_kpi_matrix=relationship_kpi_matrix)

    # build weight matrix
    matrix_size = len(listKpis) + 2
    weight_matrix = zeros(matrix_size, matrix_size)

    # build harmony search solution candidate
    path_solution_candidates: Tensor = data_service.build_harmony_search_candidate(
        harmony_search, relationship_kpi_matrix, ant_colony)

    # end build harmony search solution candidate

    # data_service.run_algorithms(path_solution_candidates, object_harmony_search)

    # run algorithms
    harmony_service = HarmonyService()
    ant_colony_service = AntColonyService(ant_colony=ant_colony)
    best_path = None
    for gen in range(object_harmony_search.max_improvisations):
        harmony_service.run_algorithm(path_solution_candidates, harmony_search)
        current_gen_best = ant_colony_service.run_algorithm(
            path_solution_candidates, len(listEmployees), object_harmony_search)

        if best_path is None:
            best_path = current_gen_best

        if current_gen_best is not None and best_path is not None and best_path['path_length'] > current_gen_best['path_length']:
            best_path = current_gen_best

        # update pheromone
        ant_colony_service.update_local_pheromone(path_solution_candidates)
        ant_colony_service.update_global_pheromone(
            path_solution_candidates, best_path_position=best_path['weight_position'], best_weight=best_path['path_length'], best_path=best_path['path'])

        print("Hoàn thành gen thứ: ", gen)

    print(best_path)

    return listKpis

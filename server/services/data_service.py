from typing import List
from requests import KpiConditionRequest
from torch import zeros, int32, Tensor, tensor, ones, pow, ones_like, zeros_like
from models import HarmonySearch, AntColony
from helpers import START_POINT_NAME, FINISH_POINT_NAME
from models import ObjectHarmonySearch


class DataService:
    def build_kpi_relationship_matrix(kpiConditions: List[KpiConditionRequest]) -> Tensor:
        matrix_size = len(kpiConditions) + 2
        matrix = zeros(matrix_size, matrix_size, dtype=int32)

        for kpi_condition in kpiConditions:
            index_kpi_col = int(kpi_condition.id)
            if index_kpi_col >= matrix_size:
                continue

            if matrix[0, index_kpi_col] == 0:
                matrix[0, index_kpi_col] = 1

            for next_point in kpi_condition.post_condition:
                next_point_index = int(next_point)
                if next_point_index >= matrix_size:
                    continue
                matrix[index_kpi_col, next_point_index] = 1

        return matrix

    def build_harmony_search_candidate(harmony_search: HarmonySearch, relationship_kpi_matrix: Tensor, ant_colony: AntColony) -> List:
        candidates: List = list()
        for row in range(relationship_kpi_matrix.size(0)):
            for col in range(relationship_kpi_matrix.size(1)):
                if relationship_kpi_matrix[row, col] > 0 and col != harmony_search.objective_harmony_search.number_parameters + 1:
                    harmony_memory = harmony_search.initialize_harmony_memory()
                    harmony_pheromone_candidate_value = ones_like(
                        harmony_memory)

                    payload = {
                        'from': START_POINT_NAME if row == 0 else FINISH_POINT_NAME if row == harmony_search.objective_harmony_search.number_parameters + 1 else str(row),
                        'to': START_POINT_NAME if col == 0 else FINISH_POINT_NAME if col == harmony_search.objective_harmony_search.number_parameters + 1 else str(col),
                        'harmony_memory': harmony_memory,
                        'harmony_pheromone_candidate_value': harmony_pheromone_candidate_value,
                    }

                    candidates.append(payload)

        return candidates

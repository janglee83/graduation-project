from typing import List
from requests import KpiConditionRequest
from torch import zeros, int32, Tensor


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

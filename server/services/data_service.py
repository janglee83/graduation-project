from typing import List
from requests import KpiConditionRequest, KpiRequest, Employees
from torch import zeros, int32, Tensor, tensor, stack
from models import HarmonySearch
# from helpers import START_POINT_NAME, FINISH_POINT_NAME
# from models import ObjectHarmonySearch
# from services.truncated_normal_service import TruncatedNormalService


class DataService:
    def build_kpi_relationship_matrix(kpiConditions: List[KpiConditionRequest]) -> Tensor:
        """
        Build a KPI relationship matrix base post condition of each kpi.

        Args:
            kpiConditions (List[KpiConditionRequest]): List of KPI condition requests including kpi_id and its post condition.

        Returns:
            Tensor: The KPI relationship matrix.
        """
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

    def build_lower_upper_matrix(listKpis: List[KpiRequest]) -> Tensor:
        """
        Build a lower_upper_matrix base lower and upper bound's value of each kpi.add()

        Args:
            listKpis (List[KpiRequest]): List of Kpi's detail

        Returns:
            Tensor: lower_upper_matrix
        """
        return tensor(list(map(lambda kpi: [kpi.lower_bound, kpi.upper_bound], listKpis)))

    def build_executive_staff_matrix(listKpis: List[KpiRequest], listEmployees: List[Employees]) -> Tensor:
        matrix = zeros(len(listEmployees), len(listKpis))

        for index, kpi in enumerate(listKpis):
            if kpi.executive_staff[0] == 'all':
                matrix[:, index] = 1
            else:
                for staff in kpi.executive_staff:
                    index_row = int(staff) - 1
                    matrix[index_row, index] = 1

        return matrix

    def build_hs_memory_candidate(harmony_search: HarmonySearch, lower_upper_matrix: Tensor, executive_staff_matrix: Tensor) -> List:
        object_hs = harmony_search.objective_harmony_search
        list_candidate = list(map(lambda _: harmony_search.initialize_harmony_memory(
            lower_upper_matrix, executive_staff_matrix), range(object_hs.hms)))
        return stack(list_candidate)

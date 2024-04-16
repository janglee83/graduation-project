from typing import List
from requests import Employees, KpiRequest
from torch import tensor, Tensor

class ObjectHarmonyService(object):
    def build_employee_score_vector(listEmployees: List[Employees]) -> Tensor:
        matrix = list()
        for employee in listEmployees:
            matrix.append(employee.point)

        return tensor(matrix)

    def build_kpi_value_vector(listKpis: List[KpiRequest]) -> Tensor:
        matrix = list()
        for kpi in listKpis:
            matrix.append(kpi.value)

        return tensor(matrix)

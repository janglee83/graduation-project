from typing import List
from requests import Employees, KpiRequest
from torch import tensor, Tensor


class ObjectHarmonyService(object):
    def build_employee_score_vector(listEmployees: List[Employees]) -> Tensor:
        return tensor(list(map(lambda employee: employee.point, listEmployees)))

    def build_kpi_value_vector(listKpis: List[KpiRequest]) -> Tensor:
        return tensor(list(map(lambda kpi: kpi.value, listKpis)))

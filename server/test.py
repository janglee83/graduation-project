import json
from fastapi import APIRouter
from typing import List
from requests import KpiRequest, KpiConditionRequest, Employees, Environments, KpiOutput, Equipment, TaskRequest
from services import DataService, ObjectHarmonyService
from models import ObjectHarmonySearch
from torch import Tensor, ones_like, stack
from models import HarmonySearch, AntColony
from services import HarmonyService, AntColonyService
from responses import TaskResponse, CoreResponse

listKpis = list()
kpiConditions = list()
listEmployees = list()
environmentsEffect = list()
kpiOutputsEffect = list()
equipmentsEffect = list()

with open('data.json') as f:
    d = json.load(f)
    print(d['listKpis'])
    listKpis = d['listKpis']
    kpiConditions = d['kpiConditions']
    listEmployees = d['listEmployees']
    environmentsEffect = d['environmentsEffect']
    kpiOutputsEffect = d['kpiOutputsEffect']
    equipmentsEffect = d['equipmentsEffect']

import json
import pandas as pd
import json
from requests import KpiRequest, KpiConditionRequest, EmployeeRequest, EnvironmentEffect, EquipmentEffect, HumanEffect, ProductEffect
from services import DataService, ObjectHarmonyService
from models import ObjectHarmonySearch
from torch import Tensor, ones_like, stack, tensor
from models import HarmonySearch, AntColony
from services import HarmonyService, AntColonyService
from responses import TaskResponse, CoreResponse, TaskStaffWeight
from pydantic import ValidationError
from helpers import write_core_responses_to_csv
import torch
import matplotlib.pyplot as plt
import pandas
from helpers import Timer
import pandas as pd

# Define a function to create instances from JSON


def create_instances_from_json(file_path, Model):
    with open(file_path) as f:
        data = json.load(f)
        instances = [Model(**item) for item in data]
    return instances


# Define file paths and corresponding model types
file_paths = {
    'listKpis': ('./human_60/data.json', KpiRequest),
    'kpiConditions': ('./human_60/data.json', KpiConditionRequest),
    'employee_data_set': ('./human_60/employee_data_set.csv', EmployeeRequest),
    'environment_effect': ('./human_60/environment_effect.json', EnvironmentEffect),
    'equipment_effect': ('./human_60/equipment_effect.json', EquipmentEffect),
    'human_effect': ('./human_60/human_effect.json', HumanEffect),
    'product_effect': ('./human_60/product_effect.json', ProductEffect)
}

# Load data and create instances
for key, (file_path, Model) in file_paths.items():
    if key == 'employee_data_set':
        data_frame_employee = pd.read_csv(file_path)
        for _, item in data_frame_employee.iterrows():
            employee_ins = Model(id=str(
                item['EmpID']), score=item['Score'], task_completion_rate=item['TaskCompletionRate'])
            listEmployees.append(employee_ins)
    else:
        instances = create_instances_from_json(file_path, Model)
        # Dynamically assign list name
        globals()[f'list{key.capitalize()}'] = instances

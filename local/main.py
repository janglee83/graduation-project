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

# START READ DATA FROM CSV AND JSON FILE

listKpis = list()
kpiConditions = list()
listEmployees = list()
listEnvironmentsEffectScore = list()
listEquipmentEffectScore = list()
listHumanEffectScore = list()
listProductEffectScore = list()


def create_instances_from_json(listItems, Model):
    instances = []
    for item in listItems:
        try:
            instance = Model(**item)
            instances.append(instance)
        except ValidationError as e:
            print(f"Validation error: {e}")
    return instances


# Read data from data.json
with open('./human_5/data.json') as f:
    d = json.load(f)

    listKpis = create_instances_from_json(
        listItems=d['listKpis'], Model=KpiRequest)
    kpiConditions = create_instances_from_json(
        listItems=d['kpiConditions'], Model=KpiConditionRequest)

# Read data from employee data set
data_frame_employee = pandas.read_csv('./human_5/employee_data_set.csv')

for index, item in data_frame_employee.iterrows():
    employee_ins = EmployeeRequest(id=str(item['EmpID']), score=item['Score'], task_completion_rate=item['TaskCompletionRate'])
    listEmployees.append(employee_ins)

# Read data from xx_effect.json file
with open('./human_5/environment_effect.json') as file:
    data = json.load(file)
    listEnvironmentsEffectScore = create_instances_from_json(listItems=data, Model=EnvironmentEffect)

with open('./human_5/equipment_effect.json') as file:
    data = json.load(file)
    listEquipmentEffectScore = create_instances_from_json(listItems=data, Model=EquipmentEffect)

with open('./human_5/human_effect.json') as file:
    data = json.load(file)
    listHumanEffectScore = create_instances_from_json(listItems=data, Model=HumanEffect)

with open('./human_5/product_effect.json') as file:
    data = json.load(file)
    listProductEffectScore = create_instances_from_json(listItems=data, Model=ProductEffect)

# END READ DATA FROM CSV AND JSON FILE

data_service = DataService

relationship_kpi_matrix: Tensor = data_service.build_kpi_relationship_matrix(
    kpiConditions)
lower_upper_matrix: Tensor = data_service.build_lower_upper_matrix(
    listKpis, listEmployees)
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

# build ant colony model
pheromone_matrix = torch.ones_like(harmony_memory)
zero_indices = (harmony_memory == 0)
pheromone_matrix[(harmony_memory == 0.0)] = 0.0

ant_colony = AntColony(number_ants=10, number_edge=number_kpis,
                       relationship_kpi_matrix=relationship_kpi_matrix, pheromone_matrix=pheromone_matrix)

# run algorithms
harmony_service = HarmonyService()
ant_colony_service = AntColonyService(ant_colony=ant_colony)
harmony_search.set_harmony_memory(harmony_memory=harmony_memory)


# Assuming you have a list of generations and their corresponding fitness values
generations = list()  # Replace with your actual generation numbers
fitness_values = list()  # Replace with your actual fitness values


best_path = None
for gen in range(object_harmony_search.max_improvisations):
    harmony_service.run_algorithm(harmony_search, lower_upper_matrix)
    current_gen_best = ant_colony_service.run_algorithm(harmony_search)

    if best_path is None:
        best_path = current_gen_best

    if current_gen_best is not None and best_path is not None and best_path['path_length'] > current_gen_best['path_length']:
        best_path = current_gen_best

    # update pheromone
    ant_colony_service.update_local_pheromone(ant_colony, listEnvironmentsEffectScore, listEquipmentEffectScore, object_harmony_search)

    if best_path is not None:
        ant_colony_service.update_global_pheromone(
            ant_colony, best_path=best_path, listHumanEffectScore=listHumanEffectScore, listProductEffectScore=listProductEffectScore, object_harmony_search=object_harmony_search)

    print("Quãng đường tìm thấy là: ",
          object_harmony_search.get_fitness(best_path['ant_weight']).item())
    print("Thế hệ:", gen)
    generations.append(gen+ 1)
    fitness_values.append(object_harmony_search.get_fitness(best_path['ant_weight']).item())


plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
plt.plot(generations, fitness_values, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness Value', fontsize=14)
plt.title('Fitness Value vs. Generation', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

response = list()
for col in range(len(object_harmony_search.kpi_weight_vector)):
    list_task = list()
    for row in range(3):
        list_staff_weight = best_path['ant_weight'][row, col]
        print(f"Task.No{row+1} weight = ", torch.sum(list_staff_weight).item())
        task_response = TaskResponse(task_id=f"Task.No{row + 1}", list_staff_weight=list())
        for index, staff_weight in enumerate(list_staff_weight):
            task_staff_weight_response = TaskStaffWeight(
                staff_id=f"Staff.No{index + 1}", weight=staff_weight)
            task_response.list_staff_weight.append(task_staff_weight_response)

        list_task.append(task_response)

    response.append(CoreResponse(kpi_id=f"KPI.No{col + 1}", tasks=list_task,
                    fitness_value=object_harmony_search.get_fitness(best_path['ant_weight'])))

write_core_responses_to_csv(response)

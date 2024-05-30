import pandas
from requests import ProductFactor, EnvironmentFactor, PRODUCT_FACTOR_TYPE, ENVIRONMENT_FACTOR_TYPE, Certificate, Major, MajorType, ResourceRequest, TaskLinkageRequest, TaskRequest
import json
from services import DataService, ObjectHarmonyService, HarmonyService, AntColonyService
from models import ObjectHarmonySearch, HarmonySearch, AntColony
import torch
from helpers import Timer
from typing import Any
import sys
import time


def mainFunction() -> tuple[float, Any, list[TaskLinkageRequest]]:
    bar_length = 50

    timer = Timer()
    timer.start()

    list_product_factor = []
    list_environment_factor = []
    list_certificate = []
    list_major = []
    list_major_type = []
    list_resource = []
    list_task_linkage = []
    list_task = []

    # Start Read the CSV file
    df_affected_factor = pandas.read_csv(
        'import_data/affected_factors_dataset.csv')

    # Filter rows for each factor type
    product_factor_rows = df_affected_factor[df_affected_factor['Type']
                                            == PRODUCT_FACTOR_TYPE]
    environment_factor_rows = df_affected_factor[df_affected_factor['Type']
                                                == ENVIRONMENT_FACTOR_TYPE]

    # Create instances for each factor type
    for index, item in product_factor_rows.iterrows():
        instance = ProductFactor(id=item['ID'], description=item['Description'])
        list_product_factor.append(instance)

    for index, item in environment_factor_rows.iterrows():
        instance = EnvironmentFactor(
            id=item['ID'], description=item['Description'])
        list_environment_factor.append(instance)

    df_certificate = pandas.read_csv('import_data/certificates.csv')

    for index, item in df_certificate.iterrows():
        instance = Certificate(
            id=item['id'], name=item['name'], abbreviation=item['abbreviation'], score=item['score'])
        list_certificate.append(instance)

    df_major = pandas.read_csv('import_data/majors.csv')

    for index, item in df_major.iterrows():
        instance = Major(id=item['id'], name=item['name'],
                        abbreviation=item['abbreviation'], score=item['score'])
        list_major.append(instance)

    df_major_type = pandas.read_csv('import_data/type_majors.csv')

    for index, item in df_major_type.iterrows():
        instance = MajorType(id=item['id'], name=item['name'],
                            abbreviation=item['abbreviation'], score=item['score'])
        list_major_type.append(instance)

    df_resource = pandas.read_csv('import_data/resources.csv')

    for index, item in df_resource.iterrows():
        id = item['Resource_Id']
        type = item['Type']
        description = item['Description']
        qualifications = item['Qualifications']
        experience = item['Experience']

        payload_qualification = {
            'certificate': list(),
        }

        for qualification in json.loads(qualifications):
            for key, value in qualification.items():
                if key == 'Certificate':
                    payload_qualification['certificate'].append(value)
                if key == 'Major':
                    payload_qualification['major'] = value
                else:
                    payload_qualification['major_type'] = value

        list_exp = list()
        for exp in json.loads(experience):
            list_exp.append(exp)

        instance = ResourceRequest(id=id, type=type, description=description,
                                qualifications=payload_qualification, experience=list_exp)
        list_resource.append(instance)

    df_task_linkage = pandas.read_csv('import_data/performance_task_linkage.csv')

    for index, item in df_task_linkage.iterrows():
        kpi_metric_id = item['KPI_Metric_Id']
        task_id = item['Task_ID']
        resource_ids = json.loads(item['Resource_IDs'])
        duration_resource_ids = json.loads(item['Durations_Resource_IDs'])
        task_weight = item['Task_Weight']

        instance = TaskLinkageRequest(kpi_metric_id=kpi_metric_id, task_id=task_id, resource_ids=resource_ids,
                                    duration_resource_ids=duration_resource_ids, task_weight=task_weight)
        list_task_linkage.append(instance)

    df_task = pandas.read_csv('import_data/tasks.csv')

    for index, item in df_task.iterrows():
        task_id = item['task_id']
        task_type = item['task_type']
        description = item['description']
        pre_tasks = json.loads(item['pre_tasks'])
        requirements = json.loads(item['requirements'])
        affected_factors = json.loads(item['affected_factors'])

        instance = TaskRequest(task_id=task_id, task_type=task_type, description=description, pre_tasks=pre_tasks, requirements=requirements, affected_factors=affected_factors)

        list_task.append(instance)

    # End Read the CSV file

    data_service = DataService()
    relationship_task_tensor = data_service.build_task_relationship(list_task)
    lower_upper_matrix = data_service.build_lower_upper_matrix(list_resource, list_task_linkage)

    # all resource can do that task

    # define const
    number_task_each = 3
    number_task = list_task_linkage[-1].task_id
    number_kpi = list_task_linkage[-1].kpi_metric_id
    number_resource = list_resource[-1].id
    number_ant = 5

    # Start build Harmony search instance
    object_harmony_service = ObjectHarmonyService
    task_kpi_weight_vector = object_harmony_service.build_task_kpi_weight_vector(
        list_task_linkage, number_kpi, number_task_each)

    object_harmony_search = ObjectHarmonySearch(
        number_parameters=number_resource, task_kpi_weight_vector=task_kpi_weight_vector, lower_upper_matrix=lower_upper_matrix)
    harmony_search = HarmonySearch(
        objective_harmony_search=object_harmony_search)

    # build harmony search solution candidate
    harmony_memory: torch.Tensor = data_service.build_hs_memory_candidate(harmony_search, lower_upper_matrix, number_task_each, number_kpi, number_resource)

    # build ant colony model
    pheromone_matrix = torch.ones(object_harmony_search.hms, number_task_each, number_kpi)
    duration_matrix = data_service.build_duration_matrix(list_task_linkage, number_task_each, number_kpi, number_resource)

    ant_colony = AntColony(number_ants=number_ant, number_edge=number_task,
                        relationship_kpi_matrix=relationship_task_tensor, pheromone_matrix=pheromone_matrix, duration_matrix=duration_matrix)

    # START RUN ALGORITHMS
    harmony_service = HarmonyService()
    ant_colony_service = AntColonyService(ant_colony=ant_colony)
    harmony_search.set_harmony_memory(harmony_memory=harmony_memory)

    # Assuming you have a list of generations and their corresponding fitness values
    generations = list()
    fitness_values = list()
    best_path = None

    for gen in range(object_harmony_search.max_improvisations):
        harmony_service.run_algorithm(harmony_search, lower_upper_matrix, number_task_each, number_kpi, number_resource)
        current_gen_best = ant_colony_service.run_algorithm(harmony_search, number_task_each, number_kpi, number_resource, list_task_linkage, number_task)

        if best_path is None:
            best_path = current_gen_best

        if current_gen_best is not None and best_path is not None and best_path['path_length'] > current_gen_best['path_length']:
            best_path = current_gen_best

        ant_colony_service.update_local_pheromone(ant_colony, listTask=list_task, listTaskLinkage=list_task_linkage, num_row=number_task_each, num_col=number_kpi)

        if best_path is not None:
            ant_colony_service.update_global_pheromone(
                ant_colony, best_path=best_path, listTask=list_task, listTaskLinkage=list_task_linkage, num_row=number_task_each, num_col=number_kpi)

        progress = (gen + 1) / object_harmony_search.max_improvisations
        bar = '#' * int(progress * bar_length)
        sys.stdout.write('\r[{:<50}] {:.2f}%'.format(bar, progress * 100))
        sys.stdout.flush()
        time.sleep(0.0000005)  # Adjust speed here

    timer.end()

    return timer.get_runtime(), best_path, list_task_linkage

import pandas
from requests import ProductFactor, EnvironmentFactor, PRODUCT_FACTOR_TYPE, ENVIRONMENT_FACTOR_TYPE, Certificate, Major, MajorType, ResourceRequest, TaskRequest, MetricRequest, TaskAssignRequest, TaskExpRequest, ExpInstance, EnvironmentTaskScore, ProductTaskScore
from services import DataService, HarmonyService, AntColonyService
from models import ObjectHarmonySearch, HarmonySearch, AntColony
import torch
import os


def calculate_prob_transit(harmony_search: HarmonySearch, pheromoneMatrix: torch.Tensor, durationMatrix: torch.Tensor):
    probabilities = ((1 / torch.sum(harmony_search.harmony_memory, dim=3)) ** 0.6) * (
        pheromoneMatrix ** 0.4) * ((1 / durationMatrix) ** (0.6))

    total = torch.sum(probabilities)
    prob_tensor = probabilities / total

    return prob_tensor


def generate_rho_matrix(num_row: int, num_col: int, listEnvironmentTaskScore: list[EnvironmentTaskScore], listTasks: list[TaskRequest], listProductTaskScore: list[ProductTaskScore], isGlobal: bool):
    matrix = torch.zeros(num_row, num_col)

    task_env_dict = {
        item.task_id: item.mean_score for item in listEnvironmentTaskScore}

    task_prod_dict = {
        item.task_id: item.mean_score for item in listProductTaskScore}

    for task in listTasks:
        task_id = task.task_id
        task_index = (task_id - 1) % num_row
        metric_index = task.metric_id - 1
        if isGlobal:
            matrix[task_index, metric_index] = torch.tensor(
                task_env_dict[task_id])
        else:
            matrix[task_index, metric_index] = torch.tensor(
                task_prod_dict[task_id])

    return matrix / 5


def read_affected_factor_dataset():
    list_product_factor = list()
    list_environment_factor = list()

    df = pandas.read_csv('import_data/affected_factors_dataset.csv')

    # Filter rows for each factor type
    product_factor_rows = df[df['Type'] == PRODUCT_FACTOR_TYPE]
    environment_factor_rows = df[df['Type'] == ENVIRONMENT_FACTOR_TYPE]

    # Create instances for each factor type
    for _, item in product_factor_rows.iterrows():
        instance = ProductFactor(
            id=item['ID'], description=item['Description'])
        list_product_factor.append(instance)

    for _, item in environment_factor_rows.iterrows():
        instance = EnvironmentFactor(
            id=item['ID'], description=item['Description'])
        list_environment_factor.append(instance)

    return list_product_factor, list_environment_factor


def read_certificates_dataset():
    list_certificate = list()

    df = pandas.read_csv('import_data/certificates.csv')

    for _, item in df.iterrows():
        instance = Certificate(
            id=item['ID'], name=item['Name'], abbreviation=item['Abbreviation'], score=item['Score'])
        list_certificate.append(instance)

    return list_certificate


def read_majors_dataset():
    list_major = list()

    df = pandas.read_csv('import_data/majors.csv')

    for _, item in df.iterrows():
        instance = Major(id=item['ID'], name=item['Name'],
                         abbreviation=item['Abbreviation'], score=item['Score'])
        list_major.append(instance)

    return list_major


def read_major_type_dataset():
    list_major_type = list()

    df = pandas.read_csv('import_data/type_majors.csv')

    for _, item in df.iterrows():
        instance = MajorType(id=item['ID'], name=item['Name'],
                             abbreviation=item['Abbreviation'], score=item['Score'])
        list_major_type.append(instance)

    return list_major_type


def read_resource_dataset(listCertificate: list[Certificate], listMajor: list[Major], listMajorType: list[MajorType]):
    list_resource = []

    # Read the CSV file into a DataFrame
    df = pandas.read_csv('import_data/resources.csv')

    # Create dictionaries for quick lookup
    cert_dict = {cert.abbreviation: cert.score for cert in listCertificate}
    major_dict = {major.abbreviation: major.score for major in listMajor}
    major_type_dict = {
        major_type.abbreviation: major_type.score for major_type in listMajorType}

    for _, item in df.iterrows():
        id = item['ID']
        type = 'Human'
        code = item['Ms']
        name = item['Name']
        description = item['Description']
        certificates_str: str = item['Certificates']
        major = item['Major']
        major_type = item['MajorType']

        certificates = certificates_str.strip("[]").split(",")

        # Calculate certification score
        certi_scores = [cert_dict.get(certi, 0) for certi in certificates]
        certi_score_mean = sum(certi_scores) / \
            (5 * len(certificates)) if certificates else 0
        major_score_mean = major_dict.get(major, 0) / 5
        major_type_score_mean = major_type_dict.get(major_type, 0) / 5
        resource_score = (certi_score_mean +
                          major_score_mean + major_type_score_mean) / 3

        instance = ResourceRequest(
            id=id, type=type, code=code, name=name, description=description,
            certificates=certificates, major=major, major_type=major_type, score=resource_score
        )
        list_resource.append(instance)

    return list_resource


def read_metrics_dataset():
    list_metrics = list()

    df = pandas.read_csv('import_data/metrics_cs_dataset.csv')

    for _, item in df.iterrows():
        id = item['ID']
        description = item['Description']
        target = item['Target']
        unit = item['Unit']
        weight = item['Weight']
        list_of_task: str = item['Listoftask']

        trimmed_string = list_of_task.strip("[]").split(",")
        list_of_task = [int(item) for item in trimmed_string]

        instance = MetricRequest(id=id, description=description, target=target,
                                 unit=unit, weight=weight, list_of_task=list_of_task)
        list_metrics.append(instance)

    return list_metrics


def read_task_dataset():
    list_task = list()

    df = pandas.read_csv('import_data/tasks_cs_dataset.csv')

    for _, item in df.iterrows():
        task_id = item['TaskID']
        description = item['Description']
        metric_id = item['Metric_ID']
        value = item['Value']
        unit = item['Unit']
        weight = item['Weight']
        start_date = item['StartDate']
        end_date = item['EndDate']
        duration = item['Duration']

        instance = TaskRequest(task_id=task_id, description=description, metric_id=metric_id, value=value,
                               unit=unit, weight=weight, start_date=start_date, end_date=end_date, duration=duration)

        list_task.append(instance)

    return list_task


def read_task_assign_dataset():
    list_task_assign = list()

    df = pandas.read_csv('import_data/task_assign_cs_dataset.csv')

    for _, item in df.iterrows():
        task_id = item['TaskId']
        resource_ids: str = item['resource_ids']
        metric_id = item['metric_id']

        trimmed_string = resource_ids.strip("[]").split(",")
        resource_ids = [int(item) for item in trimmed_string]

        instance = TaskAssignRequest(
            task_id=task_id, resource_ids=resource_ids, metric_id=metric_id)

        list_task_assign.append(instance)

    return list_task_assign


def read_exp_task_dataset(listResource: list[ResourceRequest]):
    list_task_exp = []

    df = pandas.read_csv('import_data/exp_task_dataset.csv')

    resource_dict = {resource.code: resource.id for resource in listResource}

    for _, item in df.iterrows():
        task_id = item['Task_ID']
        list_resource_value = [
            ExpInstance(resource_id=resource_dict.get(
                column, 0), value=item[column])
            for column in df.columns if column != 'Task_ID'
        ]

        instance = TaskExpRequest(
            task_id=task_id, list_exp=list_resource_value)
        list_task_exp.append(instance)

    return list_task_exp


def read_environment_factor_score_dataset():
    list_environment_task_score = list()

    df = pandas.read_csv('import_data/environment_factor_task_cs.csv')

    df['mean'] = df.drop(columns=['Task_ID']).mean(axis=1)

    for _, item in df.iterrows():
        task_id = item['Task_ID']
        mean_score = item['mean']

        instance = EnvironmentTaskScore(task_id=task_id, mean_score=mean_score)

        list_environment_task_score.append(instance)

    return list_environment_task_score


def read_product_factor_score_dataset():
    list_product_task_score = list()

    df = pandas.read_csv('import_data/product_factor_task_cs.csv')

    df['mean'] = df.drop(columns=['Task_ID']).mean(axis=1)

    for _, item in df.iterrows():
        task_id = item['Task_ID']
        mean_score = item['mean']

        instance = ProductTaskScore(task_id=task_id, mean_score=mean_score)

        list_product_task_score.append(instance)

    return list_product_task_score


def mainFunction():
    list_product_factor, list_environment_factor = read_affected_factor_dataset()
    list_certificate = read_certificates_dataset()
    list_major = read_majors_dataset()
    list_major_type = read_major_type_dataset()
    list_resource = read_resource_dataset(
        list_certificate, list_major, list_major_type)
    list_metrics = read_metrics_dataset()
    list_task = read_task_dataset()
    list_task_assign = read_task_assign_dataset()
    list_task_exp = read_exp_task_dataset(list_resource)
    list_environment_task_score = read_environment_factor_score_dataset()
    list_product_task_score = read_product_factor_score_dataset()

    data_service = DataService()

    # define const
    num_row = data_service.find_number_row(list_metrics)
    num_col = len(list_metrics)
    num_item = data_service.find_number_item(list_task_assign)
    num_task = len(list_task)
    num_improvisations = 2
    hms = 2
    num_ant = 5

    rho_local = generate_rho_matrix(num_row=num_row, num_col=num_col, listEnvironmentTaskScore=list_environment_task_score,
                                    listTasks=list_task, listProductTaskScore=list_product_task_score, isGlobal=False)
    rho_global = generate_rho_matrix(num_row=num_row, num_col=num_col, listEnvironmentTaskScore=list_environment_task_score,
                                     listTasks=list_task, listProductTaskScore=list_product_task_score, isGlobal=True)

    # generate tensor
    lower_upper_tensor = data_service.build_lower_upper_tensor(
        list_task_exp, list_resource, list_task_assign, list_metrics, num_row, num_col)
    object_harmony_search = ObjectHarmonySearch(
        lower_upper_matrix=lower_upper_tensor, max_improvisations=num_improvisations, hms=hms)
    harmony_search = HarmonySearch(
        objective_harmony_search=object_harmony_search)

    # build harmony search solution candidate
    harmony_memory: torch.Tensor = data_service.build_hs_memory_candidate(
        harmony_search, lower_upper_tensor, num_row, num_col, num_item, list_task_assign)

    # build ant colony model
    pheromone_matrix = torch.ones(
        object_harmony_search.hms, num_row, num_col)
    duration_matrix = data_service.build_duration_matrix(
        list_task, num_row, num_col)

    ant_colony = AntColony(
        number_ants=num_ant, pheromone_matrix=pheromone_matrix, duration_matrix=duration_matrix)

    # START RUN ALGORITHMS
    harmony_service = HarmonyService()
    ant_colony_service = AntColonyService(ant_colony=ant_colony)
    harmony_search.set_harmony_memory(harmony_memory=harmony_memory)

    # Assuming you have a list of generations and their corresponding fitness values
    list_solution = list()
    list_fitness = list()

    for _ in range(num_improvisations):
        harmony_service.run_algorithm(
            harmony_search, lower_upper_tensor, num_row, num_col, num_item)

        prob_transit = calculate_prob_transit(
            harmony_search=harmony_search, pheromoneMatrix=pheromone_matrix, durationMatrix=duration_matrix)

        weight, position, fitness = ant_colony_service.run_algorithm(
            harmony_search=harmony_search, num_row=num_row, num_col=num_col, num_item=num_item, hms=hms, prob_transit=prob_transit)

        gen_solution = {
            'weight': weight,
            'position': position,
            'fitness': fitness
        }

        list_solution.append(gen_solution)
        list_fitness.append(fitness)

        ant_colony_service.update_local_pheromone(
            ant_colony=ant_colony, prob_transit=prob_transit, rho_local=rho_local)

        ant_colony_service.update_global_pheromone(
            prob_transit=prob_transit, rho_global=rho_global, positions=position, fitness=fitness)

        print(fitness)

    if list_solution:
        best_solution = min(list_solution, key=lambda x: x['fitness'])

    return best_solution, torch.tensor(list_fitness)


if __name__ == "__main__":
    best_solution, list_fitness = mainFunction()

    def write_fitness_into_csv(listFitness: list):
        convert = list()
        for index, item in enumerate(listFitness):
            convert.append((index, item))

        df = pandas.DataFrame(convert, columns=['id', 'fitness'])
        file_path = 'results/fitness_ev.csv'

        if os.path.exists(file_path):
            # If file exists, find the next available filename with an incremented index
            index = 1
            while True:
                new_file_path = f"{file_path[:-4]}_{index}.csv"
                if not os.path.exists(new_file_path):
                    file_path = new_file_path
                    break
                index += 1

        df.to_csv(file_path, index=False)

    write_fitness_into_csv(list_fitness.tolist())

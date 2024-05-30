import pandas
from requests import EnterpriseGoalRequest, UnitRequestResource
from services import DataService, HarmonyService, AntColonyService
from models import ObjectHarmonySearch, HarmonySearch, AntColony
import torch
import os


def calculate_prob_transit(harmony_search: HarmonySearch, pheromoneMatrix: torch.Tensor):
    probabilities = ((1 / harmony_search.harmony_memory) ** 2) * (pheromoneMatrix ** 0.4)

    total = torch.sum(probabilities)
    prob_tensor = probabilities / total

    return prob_tensor


def generate_rho_matrix(num_row: int, num_col: int, listUnitResource: list[UnitRequestResource], isGlobal: bool):
    matrix = torch.zeros(num_row, num_col)

    for row in range(num_row):
        for col in range(num_col):
            matrix[row, col] = listUnitResource[row].employee_score

    return matrix / 100


def read_enterprise_goal_dataset():
    list_enterprise_goal = list()

    df = pandas.read_csv('import_data/enterprise_goal.csv')

    for index, row in df.iterrows():
        instance = EnterpriseGoalRequest(id=row['ID'], description=row['Description'], planed_value=row['Planed value'],
                                         unit=row['Unit'], success_criteria=row['Success criteria'])
        list_enterprise_goal.append(instance)

    return list_enterprise_goal


def read_unit_resource_dataset():
    list_unit_resource = list()

    df = pandas.read_csv('import_data/unit_resource.csv')

    for index, row in df.iterrows():
        instance = UnitRequestResource(id=index + 1, unit=row['Unit'], num_employee=row['NumEmployee'],
                                       employee_score=row['EmployeeScore'], min_score=row['minEmployeeScore'],
                                       max_score=row['maxEmployeeScore'])
        list_unit_resource.append(instance)

    return list_unit_resource


def main():
    global best_solution
    list_enterprise_goal = read_enterprise_goal_dataset()
    list_unit_resource = read_unit_resource_dataset()

    data_service = DataService()

    # define const
    num_row = len(list_unit_resource)
    num_col = len(list_enterprise_goal)
    num_improvisations = 5000
    hms = 100
    num_ant = 5

    rho_local = generate_rho_matrix(num_row=num_row, num_col=num_col, listUnitResource=list_unit_resource,
                                    isGlobal=False)
    rho_global = generate_rho_matrix(num_row=num_row, num_col=num_col, listUnitResource=list_unit_resource,
                                     isGlobal=False)

    # generate tensor
    lower_upper_tensor = data_service.build_lower_upper_tensor(num_row, listUnitResource=list_unit_resource)
    object_harmony_search = ObjectHarmonySearch(
        lower_upper_matrix=lower_upper_tensor, max_improvisations=num_improvisations, hms=hms)
    harmony_search = HarmonySearch(
        objective_harmony_search=object_harmony_search)

    # build harmony search solution candidate
    harmony_memory: torch.Tensor = data_service.build_hs_memory_candidate(harmony_search, lower_upper_tensor, num_row,
                                                                          num_col)

    # build ant colony model
    pheromone_matrix = torch.ones(object_harmony_search.hms, num_row, num_col)

    ant_colony = AntColony(number_ants=num_ant, pheromone_matrix=pheromone_matrix)

    # START RUN ALGORITHMS
    harmony_service = HarmonyService()
    ant_colony_service = AntColonyService(ant_colony=ant_colony)
    harmony_search.set_harmony_memory(harmony_memory=harmony_memory)

    # Assuming you have a list of generations and their corresponding fitness values
    list_solution = list()
    list_fitness = list()
    current_best_solution = None
    current_best_fitness = None
    current_best_position = None

    for _ in range(num_improvisations):
        harmony_service.run_algorithm(harmony_search, lower_upper_tensor, num_row, num_col)

        prob_transit = calculate_prob_transit(harmony_search=harmony_search, pheromoneMatrix=pheromone_matrix)

        weight, position, fitness = ant_colony_service.run_algorithm(harmony_search=harmony_search, num_row=num_row,
                                                                     num_col=num_col, prob_transit=prob_transit)

        if current_best_solution is None:
            current_best_solution = weight
            current_best_fitness = fitness
            current_best_position = position

        if current_best_solution is not None and current_best_fitness > fitness:
            current_best_solution = weight
            current_best_fitness = fitness
            current_best_position = position

        gen_solution = {
            'weight': weight,
            'position': position,
            'fitness': fitness
        }

        list_solution.append(gen_solution)
        list_fitness.append(current_best_fitness)

        ant_colony_service.update_local_pheromone(ant_colony=ant_colony, prob_transit=prob_transit, rho_local=rho_local)

        ant_colony_service.update_global_pheromone(prob_transit=prob_transit, rho_global=rho_global, positions=position,
                                                   fitness=fitness)

        print(current_best_fitness, current_best_solution.sum(dim=0))

    return torch.tensor(list_fitness), pheromone_matrix, current_best_solution, current_best_fitness


if __name__ == "__main__":
    list_fitness, pheromone_matrix, best_solution, best_fitness = main()

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

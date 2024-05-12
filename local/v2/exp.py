import main
import pandas
import torch
from requests import TaskLinkageRequest

def write_runtime_into_csv(listRunTime: list):
    function_runtimes = [(i + 1, runtime)
                         for i, runtime in enumerate(listRunTime)]
    df = pandas.DataFrame(function_runtimes, columns=['id', 'runtime'])
    df.to_csv('results/runtime.csv', index=False)


def write_best_fitness_into_csv(listBestFitness: list):
    function_runtimes = [(i + 1, runtime)
                         for i, runtime in enumerate(listBestFitness)]
    df = pandas.DataFrame(function_runtimes, columns=['id', 'runtime'])
    df.to_csv('results/fitness.csv', index=False)


def write_solution_into_csv(solution: torch.Tensor, list_task_linkage_global: list[TaskLinkageRequest]):
    for task in list_task_linkage_global:
        kpi_index = task.kpi_metric_id - 1
        task_index = (task.task_id - 1) % 3

        task.resource_weight = solution[task_index, kpi_index, :].tolist()

    data = [{k: v for k, v in item.model_dump().items() if k != 'duration_resource_ids'}
            for item in list_task_linkage_global]

    column_names = [k for k in TaskLinkageRequest.__annotations__.keys()
                    if k != 'duration_resource_ids']

    df = pandas.DataFrame(data, columns=column_names)

    df.to_csv('results/solution.csv', index=False)

if __name__ == "__main__":
    time_repeat_func = 20

    list_runtime = list()
    tensor_best_weight = []
    list_best_path = list()
    best_fitness = float('-inf')
    log_list_best_fitness = list()
    list_task_linkage_global: list[TaskLinkageRequest]

    for i in range(time_repeat_func):
        runtime, best_path, list_task_linkage = main.mainFunction()

        list_task_linkage_global = list_task_linkage

        # log run time
        list_runtime.append(runtime)

        # log result
        fitness = best_path['path_length']
        log_list_best_fitness.append(fitness.item())

        if best_fitness == float('-inf'):
            best_fitness = fitness

        if best_fitness >= fitness:
            tensor_best_weight = best_path['ant_weight']
            list_best_path = best_path['path_weight']

    write_runtime_into_csv(list_runtime)
    write_best_fitness_into_csv(log_list_best_fitness)
    write_solution_into_csv(tensor_best_weight, list_task_linkage_global)



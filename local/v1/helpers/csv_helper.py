import csv
from typing import List
import os

def flatten_core_responses(core_responses: List) -> List[dict]:
    flat_data = []
    for core_response in core_responses:
        for task in core_response.tasks:
            for staff_weight in task.list_staff_weight:
                flat_task = {
                    "kpi_id": core_response.kpi_id,
                    "fitness_value": core_response.fitness_value,
                    "task_id": task.task_id,
                    "staff_id": staff_weight.staff_id,
                    "weight": staff_weight.weight
                }
                flat_data.append(flat_task)
    return flat_data


def write_core_responses_to_csv(core_responses: List, file_path: str="core_responses.csv"):
    flat_data = flatten_core_responses(core_responses)
    headers = ["kpi_id", "fitness_value", "task_id", "staff_id", "weight"]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(flat_data)

    # Check if the file already exists
    if os.path.exists(file_path):
        # If file exists, find the next available filename with an incremented index
        index = 1
        while True:
            new_file_path = f"{file_path[:-4]}_{index}.csv"
            if not os.path.exists(new_file_path):
                file_path = new_file_path
                break
            index += 1

    flat_data = flatten_core_responses(core_responses)
    headers = ["kpi_id", "fitness_value", "task_id", "staff_id", "weight"]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(flat_data)


# # Check if the file already exists
#     if os.path.exists(file_path):
#         # If file exists, find the next available filename with an incremented index
#         index = 1
#         while True:
#             new_file_path = f"{file_path[:-4]}_{index}.csv"
#             if not os.path.exists(new_file_path):
#                 file_path = new_file_path
#                 break
#             index += 1

#     flat_data = flatten_core_responses(core_responses)
#     headers = ["kpi_id", "fitness_value", "task_id", "staff_id", "weight"]

#     with open(file_path, mode='w', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=headers)
#         writer.writeheader()
#         writer.writerows(flat_data)


# write_core_responses_to_csv(core_responses, "core_responses.csv")

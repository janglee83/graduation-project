import json

with open('product_effect.json') as file:
    data = json.load(file)


# Remove the attributes you want
attributes_to_remove = ['_id']
for attr in attributes_to_remove:
    for index, item in enumerate(data):
        if attr in item:
            del data[index][attr]


# Rename the desired key (column) in each dictionary
old_key = 'employeeId'
new_key = 'employee_id'
for entry in data:
    if old_key in entry:
        entry[new_key] = entry.pop(old_key)

def process_entry(entry, kpi_id):
    list_tasks = []
    for i in range(1, 4):
        task_key = f'task {i}'
        task_score = entry['kpi 1'][task_key]
        task_dict = {
            'task_id': str(i),
            'score': task_score
        }
        list_tasks.append(task_dict)

    new_payload = {
        'kpi_id': kpi_id,
        'tasks': list_tasks
    }
    return new_payload

for entry in data:
    list_kpi = list()
    for kpi_id in ['1', '2', '3', '4', '5']:
        new_payload = process_entry(entry, kpi_id)
        list_kpi.append(new_payload)
    # print(list_kpi)
    entry['list_kpi'] = list_kpi
    del entry['kpi 1']
    del entry['kpi 2']
    del entry['kpi 3']
    del entry['kpi 4']
    del entry['kpi 5']

# Save the modified list of dictionaries back to a JSON file
with open('product_effect.json', 'w') as f:
    json.dump(data, f, indent=4)  # indent parameter for pretty formatting (optional)

import pandas
import json

# Load the CSV file into a pandas DataFrame
data_frame = pandas.read_csv('data/HRDataset_v14.csv')

# List all columns in the DataFrame
columns_list = data_frame.columns.tolist()
list_column_to_remove = list()
list_column_not_to_remove = ['EmpID', 'PerfScoreID', 'PerformanceScore']
for column in columns_list:
    if column not in list_column_not_to_remove:
        list_column_to_remove.append(column)

# remove column in data frame
data_frame.drop(list_column_to_remove, axis=1, inplace=True)

def map_scores(score_id):
    if score_id == 4:
        return 1.25
    elif score_id == 3:
        return 1
    elif score_id == 2:
        return 0.75
    elif score_id == 1:
        return 0.5
    else:
        return None  # Handle other cases if necessary

# Apply the function to create a new column
data_frame['Score'] = data_frame['PerfScoreID'].apply(map_scores)
# data_frame['EmpID'] = data_frame['EmpID'] - data_frame['EmpID'].min() + 1
data_frame['EmpID'] = range(1, len(data_frame) + 1)

# read data from json file
list_capacity = list()
with open('data/capacity.json') as f:
    d = json.load(f)
    list_capacity = d

# Step 3: Update DataFrame based on JSON data
for item in list_capacity:
    for index, data_frame_item in data_frame.iterrows():
        if data_frame_item['EmpID'] == item['EmployeeID']:
            # Update the value in the DataFrame
            data_frame.at[index, 'TaskCompletionRate'] = item['TaskCompletionRate']

print(data_frame.head(60))

# Save the modified DataFrame back to a CSV file
data_frame.head(60).to_csv('data/cleaned_data.csv', index=False)


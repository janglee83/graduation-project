import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv('./results/medium/fitness_ev.csv')

# Extracting x and y values from the DataFrame
x = data['id']
y = sorted(data['fitness'], reverse=True)

# Plotting the data with custom line style and color
plt.plot(x, y, linestyle='-')

# Adding labels and title
plt.xlabel('ID')
plt.ylabel('Fitness')
plt.title('Fitness vs ID')

# Displaying the plot
plt.show()

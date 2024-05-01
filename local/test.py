import matplotlib.pyplot as plt

# Assuming you have a list of generations and their corresponding fitness values
generations = [1, 2, 3, 4, 5]  # Replace with your actual generation numbers
fitness_values = [10, 20, 30, 25, 35]  # Replace with your actual fitness values

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

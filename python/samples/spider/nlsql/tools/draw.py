import matplotlib.pyplot as plt
import numpy as np

# Sample data
models = ['Mistral Nemo', 'Qwen 2.5', 'Gemma 3']
categories = ['Extra Hard', 'Hard', 'Medium', 'Easy']
results = [
    [0.85, 0.80, 0.82, 0.82],  # Model A
    [0.88, 0.84, 0.86, 0.86],  # Model B
    [0.83, 0.79, 0.81, 0.81]   # Model C
]
colors = ['#5DADE2',  # Mid blue
          '#F194C1',  # Mid pink
          '#A569BD']  # Mid purple

# Bar chart parameters
x = np.arange(len(categories))  # the label locations
width = 0.25  # the width of the bars

# Create subplots
fig, ax = plt.subplots()

# Draw each model's bar
for i in range(len(models)):
    ax.bar(x + i * width, results[i], width, label=models[i], color=colors[i])


# Labels and legend
ax.set_ylabel('Scores')
ax.set_title('Model Comparison by Metric')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend()

# Save to file
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
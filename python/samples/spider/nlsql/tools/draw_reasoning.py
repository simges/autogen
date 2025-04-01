import matplotlib.pyplot as plt
import numpy as np

# Sample data
models = ['Qwen2.5 14B', 'Gemma3-PT 12B', 'Gemma3-IT 12B']
categories = ['MMLU-Pro', 'HumanEval', 'MBPP', 'Big-Bench Hard', 'Math']
results = [
    [0.512, 0.567, 0.767, 0.782, 0.556],  # Qwen 2.5
    [0.453, 0.457, 0.604, 0.726, 0.433],  # Gemma 3 PT
    [0.606, 0.854, 0.730, 0.857, 0.838]   # Gemma 3 IT
]
colors = ['#1f77b4',  # blue
          '#2ca02c',  # green
          '#ff7f0e']  # orange

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
ax.set_title('Benchmarks')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend()

# Save to file
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
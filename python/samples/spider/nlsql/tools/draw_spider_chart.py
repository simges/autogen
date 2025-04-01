import matplotlib.pyplot as plt
import numpy as np

# Labels for the six dimensions
labels = ['#Examples', '#DB', '#Table/DB', '#Rows/DB', '#Keywords', '#JOINs/SQL']
num_vars = len(labels)

# Dataset
data = {
    'WikiSQL': [80654, 26521, 1,   17,     12.0,  0.0],
    'Spider':  [10181, 200,   5.1, 2000,   36.0,  0.5],
    'BIRD':    [12751, 95,    7.3, 549000, 51.0,  1.0],
}

# Color map for each dataset
colors = {
    'WikiSQL': 'blue',
    'Spider': 'green',
    'BIRD': 'red'
}

# Prepare data
dataset_names = list(data.keys())
raw_values = np.array(list(data.values()))  # shape: (3 datasets, 6 dimensions)

# Normalize each dimension (column) independently
min_vals = raw_values.min(axis=0)
max_vals = raw_values.max(axis=0)
normalized_values = (raw_values - min_vals) / (max_vals - min_vals + 1e-6)

# Compute angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the circle

# Create the radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each dataset with its specific color (without filling)
for i, name in enumerate(dataset_names):
    values = normalized_values[i].tolist()
    values += values[:1]  # close the loop
    ax.plot(angles, values, label=name, color=colors[name], linewidth=2)

# Configure chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
plt.title("Spider Chart: WikiSQL (blue), Spider (green), BIRD (red)", y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Save to file
plt.savefig("no_fill_spider_chart.png", dpi=300)
plt.close()

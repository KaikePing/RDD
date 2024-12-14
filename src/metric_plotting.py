import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the JSON file
file_path = "../results/metrics_results.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract versions and metrics
versions = list(data.keys())
metric_names = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness"
]

# Prepare the data for plotting
metrics = {version: [] for version in versions}
for version in versions:
    for metric in metric_names:
        metrics[version].append(data[version][0][metric])

# Bar chart
x = np.arange(len(metric_names))  # X-axis positions
bar_width = 0.25

plt.figure(figsize=(10, 6))

# Plot each version's metrics
for i, version in enumerate(versions):
    plt.bar(x + i * bar_width, metrics[version], width=bar_width, label=version)

# Chart details
plt.xticks(x + bar_width, [metric.split("/")[-1] for metric in metric_names])
plt.ylabel('Metric Values')
plt.title('YOLO Road Damage Detection Metrics Comparison')
plt.legend(title="YOLO Versions")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the chart to the specified directory
output_dir = "../images"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_path = os.path.join(output_dir, "yolo_metrics_comparison.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)  # Save the figure
print(f"Chart saved to {output_path}")
plt.close()

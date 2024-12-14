import os
import pandas as pd
import json
from ultralytics import YOLO
import xml.etree.ElementTree as ET

# Define paths
annotations_path = "../United_States/train/annotations/xmls"
images_path = "../United_States/train/images"
split_csv_path = "../United_States/dataset_split.csv"
labels_path = "../United_States/train/labels"
os.makedirs(labels_path, exist_ok=True)
metrics_output_path = "../results/metrics_results.json"

# Damage type mapping
damage_type_mapping = {
    "D00": 0,  # Longitudinal Cracks
    "D10": 1,  # Transverse Cracks
    "D20": 2,  # Alligator Cracks
    "D40": 3   # Potholes
}

# Read dataset split
split_df = pd.read_csv(split_csv_path)

# Convert XML annotations to YOLO format
def convert_annotation(xml_file, output_label_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_file = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    label_lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in damage_type_mapping:
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            label_lines.append(f"{damage_type_mapping[name]} {x_center} {y_center} {bbox_width} {bbox_height}\n")

    # Save label file
    if label_lines:
        label_file = os.path.join(output_label_path, os.path.splitext(image_file)[0] + ".txt")
        with open(label_file, "w") as f:
            f.writelines(label_lines)

# Process data
for _, row in split_df.iterrows():
    xml_file = os.path.join(annotations_path, row["filename"])
    convert_annotation(xml_file, labels_path)

# Create data.yaml file
data_yaml_path = "../United_States/data.yaml"
with open(data_yaml_path, "w") as f:
    f.write(f"train: {os.path.abspath(images_path)}\n")
    f.write(f"val: {os.path.abspath(images_path)}\n")
    f.write("nc: 4\n")
    f.write("names: ['Longitudinal Cracks', 'Transverse Cracks', 'Alligator Cracks', 'Potholes']\n")

# Check for existing metrics results
if os.path.exists(metrics_output_path):
    with open(metrics_output_path, "r") as f:
        try:
            metrics_results = json.load(f)
        except json.JSONDecodeError:
            metrics_results = {"v5": [], "v8": [], "v11": []}
else:
    metrics_results = {"v5": [], "v8": [], "v11": []}

# Define model versions and run counts
model_versions = {"v5": "yolov5s.pt", "v8": "yolov8s.pt", "v11": "yolo11s.pt"}
runs_per_version = 1

for version, model_path in model_versions.items():
    for run in range(runs_per_version):
        print(f"Running {version} - Run {run + 1}")
        
        # Check if this version and run is already completed
        if len(metrics_results[version]) > run:
            print(f"Skipping {version} - Run {run + 1} as it is already completed.")
            continue

        # Load model
        model = YOLO(model_path)

        # Start training
        model.train(data=data_yaml_path, epochs=100, imgsz=640, patience=10, save=True, save_period=1)

        # Evaluate on validation set
        results = model.val(data=data_yaml_path, imgsz=640)

        # Save results
        metrics_results[version].append(results.results_dict)

        # Save to file
        with open(metrics_output_path, "w") as f:
            json.dump(metrics_results, f, indent=4)

# Print final results
print("All metrics saved:")
print(metrics_results)

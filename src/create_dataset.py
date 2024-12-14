import os
import random
import xml.etree.ElementTree as ET
import pandas as pd

# Define the folder path
annotations_path = "../United_States/train/annotations/xmls"

# Get all XML files
xml_files = [f for f in os.listdir(annotations_path) if f.endswith(".xml")]

# Shuffle the file order randomly
random.seed(5834)  # Fix the random seed to ensure reproducibility
random.shuffle(xml_files)

# Split the dataset into 8:2 ratio
split_index = int(len(xml_files) * 0.8)
train_files = xml_files[:split_index]
test_files = xml_files[split_index:]

# Create DataFrame
data = []
for file_name in train_files:
    data.append({"filename": file_name, "split": "train"})
for file_name in test_files:
    data.append({"filename": file_name, "split": "test"})

df = pd.DataFrame(data)

# Save as CSV file
csv_path = "../United_States/dataset_split.csv"
df.to_csv(csv_path, index=False)

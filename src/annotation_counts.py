import os
import xml.etree.ElementTree as ET
from collections import Counter

# Define the folder path containing XML files
folder_path = "../United_States/train/annotations/xmls"

# Initialize the counter
dxx_counter = Counter()

# Iterate through all XML files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".xml"):  # Ensure only XML files are processed
        file_path = os.path.join(folder_path, file_name)
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        # Find all <object> tags and their <name> values
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name.startswith("D"):  # Only count DXX types
                dxx_counter[name] += 1

# Print the count results
print("Count results:")
for dxx, count in dxx_counter.items():
    print(f"{dxx}: {count}")
    
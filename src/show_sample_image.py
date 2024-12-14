import os
import random
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

# Define folder paths
annotations_path = "../United_States/train/annotations/xmls"
images_path = "../United_States/train/images"
output_path = "../images"

# Damage type mapping table
damage_type_mapping = {
    "D00": "Longitudinal Cracks",
    "D10": "Transverse Cracks",
    "D20": "Alligator Cracks",
    "D40": "Potholes"
}

# Randomly select 4 samples for each damage type
all_samples = {damage: [] for damage in damage_type_mapping.keys()}

# Traverse all XML files to collect samples
for file_name in os.listdir(annotations_path):
    if file_name.endswith(".xml"):
        file_path = os.path.join(annotations_path, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find all <object> tags
        for obj in root.findall("object"):
            name = obj.find("name").text
            truncated = int(obj.find("truncated").text)
            difficult = int(obj.find("difficult").text)
            
            # Only select samples with truncated=0 and difficult=0
            if name in damage_type_mapping and truncated == 0 and difficult == 0:
                all_samples[name].append(file_name)

# Random sampling, select up to 4 files for each type
random_samples = {
    damage: random.sample(files, min(4, len(files))) for damage, files in all_samples.items()
}

# Plot sample images and save
os.makedirs(output_path, exist_ok=True)

def display_samples():
    fig, axes = plt.subplots(len(random_samples), 1, figsize=(18, 5 * len(random_samples)))

    for idx, (damage, file_names) in enumerate(random_samples.items()):
        combined_image = []

        for xml_file in file_names:
            # Get the corresponding image file path
            tree = ET.parse(os.path.join(annotations_path, xml_file))
            root = tree.getroot()
            image_file = root.find("filename").text
            image_path = os.path.join(images_path, image_file)

            # Load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Draw bounding boxes for the corresponding category
            for obj in root.findall("object"):
                name = obj.find("name").text
                truncated = int(obj.find("truncated").text)
                difficult = int(obj.find("difficult").text)

                if name == damage and truncated == 0 and difficult == 0:
                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)

                    # Draw rectangle and category name
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(image, damage_type_mapping[name], (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            combined_image.append(image)

        # Concatenate images
        combined_image = cv2.hconcat(combined_image)

        # Display in Matplotlib
        axes[idx].imshow(combined_image)
        axes[idx].set_title(f"{damage_type_mapping[damage]}:", loc='left', fontsize=24, pad=20)
        axes[idx].axis("off")

    # Adjust layout and spacing
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    output_file = os.path.join(output_path, "Damage_Type_Example.png")
    plt.savefig(output_file)
    plt.close()

# Execute display
if __name__ == "__main__":
    display_samples()

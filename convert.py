import csv
import os
from PIL import Image

# Paths
image_folder = 'dataset/license_plates_detection_train'  # Update this to your CSV file
csv_path = 'dataset/Licplatesdetection_train.csv'  # Update this to your image folder
output_label_folder = 'dataset/yolo'  # Folder to save YOLO format labels
os.makedirs(output_label_folder, exist_ok=True)

# Assume one class, "license plate"
class_id = 0

# Process CSV
with open(csv_path, 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        image_name, x_min, y_min, x_max, y_max = row
        print(row)
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        
        # Load the image to get its width and height
        image_path = os.path.join(image_folder, image_name)
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Convert to YOLO format
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Write the label file
        label_path = os.path.join(output_label_folder, f"{os.path.splitext(image_name)[0]}.txt")
        with open(label_path, 'w') as label_file:
            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("Conversion completed. YOLO format labels are saved in:", output_label_folder)

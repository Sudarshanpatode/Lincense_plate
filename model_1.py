import cv2
import torch
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt')

# Set up paths
test_image_path = Path('/dataset/license_plates_detection_train')
output_path = Path('/dataset/license_plates_detection_train')
output_path.mkdir(exist_ok=True, parents=True)

# Detect and Draw Boundary
for image_file in test_image_path.glob('*.jpg'):
    # Load image
    img = cv2.imread(str(image_file))
    results = model(img)

    # Process detections
    for det in results.xyxy[0]:  # each detection
        x1, y1, x2, y2, conf, cls = det

        # Draw green rectangle around the number plate
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Save the result
    output_file = output_path / image_file.name
    cv2.imwrite(str(output_file), img)

    print(f"Processed and saved: {output_file}")

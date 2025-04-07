"""
    -*- coding: utf-8 -*-
    Time    : 2025/4/7 16:03
    Author  : LazyShark
    File    : txt可视化.py.py
"""
import os
import cv2

# Define paths
images_path = r'C:\Users\LazyShark\Desktop\123\images'  # Path to your images
annotations_path = r'C:\Users\LazyShark\Desktop\123\labels'  # Path to your annotations
output_path = r'C:\Users\LazyShark\Desktop\123\1'  # Path to save annotated images

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

class_name = ['cercospora', 'eyespot', 'healthy', 'redrot', 'wheat rust', 'yellow leaf']

# Function to draw bounding boxes
def draw_bbox(image, annotation):
    height, width, _ = image.shape
    with open(annotation, 'r') as file:
        for line in file:
            class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.strip().split())

            # Convert YOLO format to OpenCV format
            x1 = int((center_x - bbox_width / 2) * width)
            y1 = int((center_y - bbox_height / 2) * height)
            x2 = int((center_x + bbox_width / 2) * width)
            y2 = int((center_y + bbox_height / 2) * height)

            # Draw the bounding box
            color = (0, 255, 0)  # Green color for bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, str(class_name[int(class_id)]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image


# Process each image and its corresponding annotation
for image_file in os.listdir(images_path):
    image_path = os.path.join(images_path, image_file)
    annotation_file = os.path.splitext(image_file)[0] + '.txt'
    annotation_path = os.path.join(annotations_path, annotation_file)

    # Check if annotation file exists
    if os.path.exists(annotation_path):
        # Load image
        image = cv2.imread(image_path)

        # Draw bounding boxes
        annotated_image = draw_bbox(image, annotation_path)

        # Save the annotated image
        output_image_path = os.path.join(output_path, image_file)
        cv2.imwrite(output_image_path, annotated_image)

        print(f"Saved annotated image: {output_image_path}")

print("All images processed.")

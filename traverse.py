import os
import cv2

# Path to your dataset
dataset_path = "yolo_dataset"
images_dir = os.path.join(dataset_path, "images")
labels_dir = os.path.join(dataset_path, "labels")

# Get all image files
image_files = sorted(os.listdir(images_dir))

# Loop through all images
for image_file in image_files:
    # Construct full image path
    image_path = os.path.join(images_dir, image_file)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Construct corresponding label file path
    label_file_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))
    
    # Check if the label file exists
    if os.path.exists(label_file_path):
        # Read the labels
        with open(label_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Parse the label
                class_id, x_center, y_center, width, height = map(float, line.split())
                
                # Convert YOLO format to bounding box coordinates
                img_height, img_width, _ = image.shape
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)
                
                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image with bounding boxes
    cv2.imshow("Annotated Image", image)
    
    # Wait for key press to move to the next image
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()

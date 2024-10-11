import os
import cv2

# Path to your dataset
dataset_path = "/home/robotics/Videos/yolo_datasetv2"
images_dir = os.path.join(dataset_path, "images")
labels_dir = os.path.join(dataset_path, "labels")

# Get all image files
image_files = sorted(os.listdir(images_dir))

# Create a single window
window_name = "Annotated Image"
cv2.namedWindow(window_name)

# Define maximum window size (for example, 800x800)
max_width = 800
max_height = 800

# Initialize the index for traversing images
index = 0

while 0 <= index < len(image_files):  # Ensure index stays within bounds
    image_file = image_files[index]

    # Construct full image path
    image_path = os.path.join(images_dir, image_file)

    # Load the image
    image = cv2.imread(image_path)

    # Resize image if it's too large
    img_height, img_width, _ = image.shape
    if img_width > max_width or img_height > max_height:
        # Calculate aspect ratio to maintain it during resizing
        aspect_ratio = img_width / img_height
        if img_width > img_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

        # Resize the image
        image = cv2.resize(image, (new_width, new_height))

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

    # Update the window title with the current frame number
    cv2.setWindowTitle(window_name, f"Frame {index + 1}")

    # Display the image with bounding boxes
    cv2.imshow(window_name, image)

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF

    # Check if 'q' is pressed to quit
    if key == ord("q"):
        break

    # Check if 'd' is pressed to delete image and label
    elif key == ord("d"):
        # Delete the image file
        os.remove(image_path)
        print(f"Deleted image: {image_path}")

        # Delete the corresponding label file if it exists
        if os.path.exists(label_file_path):
            os.remove(label_file_path)
            print(f"Deleted label: {label_file_path}")

        # Remove the file from the list and adjust the index
        del image_files[index]
        index = max(index - 1, 0)  # Adjust index to prevent skipping after deletion

    # Check if the right arrow key ('→') is pressed to move forward
    elif key == ord("n"):  # or key == 2555904 for arrow keys in some systems
        if index < len(image_files) - 1:
            index += 1
        else:
            print("End of images, cannot go forward.")

    # Check if the left arrow key ('←') is pressed to move backward
    elif key == ord("p"):  # or key == 2424832 for arrow keys in some systems
        if index > 0:
            index -= 1
        else:
            print("Start of images, cannot go backward.")

# Clean up
cv2.destroyAllWindows()

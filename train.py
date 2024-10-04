import cv2
import os
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

# Load the YOLO model
model = YOLO("best.pt")

# Initialize the Roboflow inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="cGdhfWIqJzeSHPirFu7k"
)

# Open the video file
video_path = "Catching 80M Road .mp4"
cap = cv2.VideoCapture(video_path)

# Set a smaller window size (for display purposes)
display_width = 800
display_height = 450

frame_number = 0  # Initialize a counter for the frames

# Create directories for the dataset
os.makedirs('yolo_dataset/images', exist_ok=True)
os.makedirs('yolo_dataset/labels', exist_ok=True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Perform inference with the Roboflow API
        result = CLIENT.infer(frame, model_id="license-plate-recognition-rxg4e/2")
        predictions = result.get('predictions', [])

        # Filter predictions based on confidence threshold (60%)
        high_confidence_predictions = [pred for pred in predictions if pred['confidence'] > 0.50]

        # If there are no high-confidence predictions, skip this frame
        if not high_confidence_predictions:
            print(f"Skipping frame {frame_number}: No high-confidence objects detected.")
            frame_number += 1
            continue

        # Save the frame as an image only if high-confidence objects are detected
        image_path = f'yolo_dataset/images/frame_{frame_number}.jpg'
        cv2.imwrite(image_path, frame)

        # Create a corresponding label file in YOLO format
        label_path = f'yolo_dataset/labels/frame_{frame_number}.txt'
        with open(label_path, 'w') as label_file:
            for prediction in high_confidence_predictions:
                # Get bounding box info
                x = prediction['x']
                y = prediction['y']
                w = prediction['width']
                h = prediction['height']

                # Convert bounding box to YOLO format (normalized)
                img_height, img_width = frame.shape[:2]
                x_center_norm = x / img_width
                y_center_norm = y / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # Assuming the class ID is 0 for license plates (you can modify it based on your dataset)
                class_id = prediction['class_id']

                # Write the bounding box to the label file
                label_file.write(f"{class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n")

        frame_number += 1

        # Display the frame (optional)
        resized_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow("YOLO Inference with Bounding Box", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

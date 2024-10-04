import cv2
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

# Load the YOLO model
model = YOLO("license_plate_detector.pt")

# Initialize the Roboflow inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="cGdhfWIqJzeSHPirFu7k"
)

# Open the video file
video_path = "Riyadh City： Thursday rush hour.. to Ummal Hammam via Pr Turki st.mp4"
cap = cv2.VideoCapture(video_path)

# Set a smaller window size (for display purposes)
display_width = 800
display_height = 450

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Perform inference with the Roboflow API
        result = CLIENT.infer(frame, model_id="license-plate-recognition-rxg4e/2")
        print(result)

        # Extract predictions (bounding box information)
        predictions = result.get('predictions', [])

        # Draw bounding boxes on the frame
        for prediction in predictions:
            x = int(prediction['x'])
            y = int(prediction['y'])
            w = int(prediction['width'])
            h = int(prediction['height'])

            # Calculate top-left and bottom-right coordinates for the rectangle
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness=2)

            # Optionally, add a label with the confidence score
            confidence = prediction['confidence']
            label = f"License Plate ({confidence:.2f})"
            cv2.putText(frame, label, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Run YOLO inference on the frame (if still needed)
        results = model(frame)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot()

        # Resize the frame to fit within the display window
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Display the resized frame
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

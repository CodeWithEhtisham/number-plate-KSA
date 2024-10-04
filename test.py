import os
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("license_plate_detector.pt")

# Open the video file
video_path = "Riyadh Rush Hour Navigating Jarir Street Like a Local (Short Video) riyadh saudiarabia.mp4"
cap = cv2.VideoCapture(video_path)

# Create output directories for images and labels
output_dir = "yolo_dataset"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Get the current frame count based on existing images
existing_images = os.listdir(images_dir)
frame_count = len(existing_images)

# Variables to store ROI coordinates
roi_x1, roi_y1, roi_x2, roi_y2 = None, None, None, None
drawing = False  # Flag to indicate whether the user is drawing a bounding box

# Mouse callback function to draw bounding box
def draw_rectangle(event, x, y, flags, param):
    global roi_x1, roi_y1, roi_x2, roi_y2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_x1, roi_y1 = x, y  # Set starting point

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_x2, roi_y2 = x, y  # Update the end point

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_x2, roi_y2 = x, y  # Finalize the end point

# Create a named window and set the mouse callback
cv2.namedWindow("YOLO Inference")
cv2.setMouseCallback("YOLO Inference", draw_rectangle)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (1080, 720))
        frame_copy = frame.copy()
        # Draw the current bounding box if it's being defined
        if roi_x1 is not None and roi_y1 is not None and roi_x2 is not None and roi_y2 is not None:
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)  # Draw the ROI

        # Run YOLO inference on the frame
        results = model.predict(frame, conf=0.60, imgsz=640, device='cuda:0')

        # Check for detected objects
        if results[0].boxes:
            # Prepare the annotation data
            annotations = []
            detected_within_roi = False  # Flag to check if any detected box is within the ROI
            
            for box in results[0].boxes:
                # Get box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  # Assuming class IDs are integers

                # Check if the detected box is within the ROI
                if (roi_x1 is not None and roi_y1 is not None and roi_x2 is not None and roi_y2 is not None and
                        x1 >= roi_x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2):
                    detected_within_roi = True  # At least one bounding box is within the ROI
                    
                    # Calculate the YOLO format bounding box
                    img_width, img_height = frame.shape[1], frame.shape[0]
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # Append the annotation to the list
                    annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

                    # Draw the bounding box on the frame for visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the detected bounding box
                    # Add text to indicate detection
                    cv2.putText(frame, f"Plate {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Only save if we have detected objects within the ROI
            if detected_within_roi and annotations:
                # Construct filenames
                image_filename = f"{frame_count:06d}.jpg"
                label_filename = f"{frame_count:06d}.txt"
                
                # Save the frame without bounding boxes
                cv2.imwrite(os.path.join(images_dir, image_filename), frame_copy)
                
                # Save the annotations to a text file
                with open(os.path.join(labels_dir, label_filename), "w") as f:
                    f.write("\n".join(annotations))
                
                frame_count += 1  # Increment frame count for the next image

        # Display the frame with the drawn bounding boxes
        cv2.imshow("YOLO Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

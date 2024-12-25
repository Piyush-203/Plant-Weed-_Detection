

from ultralytics import YOLO
import cv2

# Constants (initially estimated)
KNOWN_DISTANCE = 30.0  # Known distance to the object (in cm)
KNOWN_DIAMETER = 10.0  # Known diameter of the object (in cm)
FOCAL_LENGTH = None  # Will be calculated

def calculate_focal_length(known_distance, known_diameter, pixel_width):
    return (pixel_width * known_distance) / known_diameter

def calculate_distance(real_diameter, focal_length, pixel_width):
    return (real_diameter * focal_length) / pixel_width

# Load the YOLOv8 model
model = YOLO('C:/Users/Dell/OneDrive/Desktop/Projects/model/weights/best.pt')  

# Capture a frame for focal length calculation
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Perform inference on the frame to get the bounding box of the known object
results = model(frame) 

# Assuming the object is detected
for result in results:
    if result.boxes:
        # Extract the width of the bounding box for the known object+
        x1, y1, x2, y2 = result.boxes[0].xyxy[0].cpu().numpy()
        pixel_width = x2 - x1

        # Calculate the focal length
        FOCAL_LENGTH = calculate_focal_length(KNOWN_DISTANCE, KNOWN_DIAMETER, pixel_width)
        print(f"Calculated Focal Length: {FOCAL_LENGTH:.2f} pixels")
        break

# Now use the calculated focal length for distance measurement
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame, conf=0.75) 

    for result in results:
        if result.boxes:
            for box in result.boxes:
                # Extract the width of the bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pixel_width = x2 - x1

                # Calculate the distance
                distance = calculate_distance(KNOWN_DIAMETER, FOCAL_LENGTH, pixel_width)

                # Draw the bounding box and distance on the frame
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                frame = cv2.putText(frame, f'Distance: {distance:.2f} cm', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display the frame with detections
    cv2.imshow('YOLOv8 Inference', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO

# --- Load calibration data ---
data = np.load("calib_data.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# Load YOLOv8 model (you can use yolov8n.pt for speed, yolov8s.pt for accuracy)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Enlarge output window
cv2.namedWindow("YOLO Distance Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Distance Estimation", 1280, 720)

# Known real-world width of object (example: person shoulder width ~0.45 m)
KNOWN_WIDTH = 0.45
FOCAL_LENGTH = camera_matrix[0, 0]  # fx from calibration matrix

def estimate_distance(perceived_width):
    if perceived_width <= 0:
        return None
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
            cls = int(box.cls[0])  # class id
            conf = float(box.conf[0])  # confidence
            label = model.names[cls]

            bbox_width = x2 - x1
            distance = estimate_distance(bbox_width)

            color = (255, 255, 255)
            status = "UNKNOWN"
            if distance:
                if distance < 1.0:
                    status, color = "CRITICAL", (0, 0, 255)
                elif distance < 2.0:
                    status, color = "WARNING", (0, 165, 255)
                elif distance >= 3.0:
                    status, color = "SAFE", (0, 255, 0)
                else:
                    status, color = "CAUTION", (255, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Put label + distance
            if distance:
                cv2.putText(frame,
                            f"{label} {conf:.2f} | {status} {distance:.2f}m",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

    cv2.imshow("YOLO Distance Estimation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

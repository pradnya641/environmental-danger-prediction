
import cv2
from agents.detection_agent import DetectionAgent

# Load YOLO and create a simple test image.
agent = DetectionAgent()
frame = cv2.imread("test_image.jpg")

if frame is None:
    raise FileNotFoundError(
        "Place a test image named test_image.jpg in the project root."
    )

detections = agent.detect_frame(frame)

for detection in detections:
    print(detection)
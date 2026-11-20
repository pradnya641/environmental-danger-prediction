
from pathlib import Path
from ultralytics import YOLO


class DetectionAgent:
    """Detect objects in video frames using a YOLO model."""

    # COCO class IDs:
    # person = 0, bicycle = 1, car = 2, motorcycle = 3,
    # bus = 5, train = 6, truck = 7
    RISK_RELEVANT_CLASSES = [0, 1, 2, 3, 5, 6, 7]

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        risk_classes_only: bool = False,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.risk_classes_only = risk_classes_only

    def detect_frame(self, frame):
        """Return filtered detections for one OpenCV frame."""

        prediction_options = {
            "source": frame,
            "conf": self.confidence_threshold,
            "verbose": False,
        }

        if self.risk_classes_only:
            prediction_options["classes"] = self.RISK_RELEVANT_CLASSES

        results = self.model.predict(**prediction_options)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

    def detect_video(
        self,
        video_path: str,
        output_path: str | None = None,
    ):
        """Process a video and optionally save an annotated copy."""
        import cv2

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")

        writer = None
        all_detections = []

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 25.0

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                detections = self.detect_frame(frame)
                all_detections.append(detections)

                for item in detections:
                    x1, y1, x2, y2 = item["bbox"]
                    label = (
                        f'{item["class_name"]} '
                        f'{item["confidence"]:.2f}'
                    )

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, label,
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2
                    )

                if output_path and writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise ValueError(
                            f"Could not create output: {output_path}"
                        )

                if writer is not None:
                    writer.write(frame)

        finally:
            capture.release()
            if writer is not None:
                writer.release()

        return all_detections


import cv2
import time
import json
from datetime import datetime

from pipeline.graph import analyze_image


CAMERA_INDEX = 0
ANALYSIS_INTERVAL = 3.0
WINDOW_NAME = "Environmental Danger Prediction System"


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 55)
    print("ENVIRONMENTAL DANGER PREDICTION SYSTEM")
    print("=" * 55)
    print("Camera started.")
    print("Gemini analysis interval:", ANALYSIS_INTERVAL, "seconds")
    print("Press Q to stop.")
    print("=" * 55)

    last_analysis_time = 0
    last_result = None
    processing = False

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Unable to read webcam frame.")
                break

            current_time = time.time()

            # Run Gemini and LangGraph periodically.
            if (
                current_time - last_analysis_time
                >= ANALYSIS_INTERVAL
                and not processing
            ):
                processing = True
                last_analysis_time = current_time

                print("\nAnalyzing camera frame...")

                try:
                    result = analyze_image(frame.copy())

                    if result.get("error"):
                        last_result = {
                            "risk_level": "UNCERTAIN",
                            "status": "MANUAL_REVIEW",
                            "explanation": result["error"],
                            "recommendations": [
                                "Check camera visibility and Gemini connectivity."
                            ]
                        }
                    else:
                        last_result = result["risk_result"]

                    print("\nRisk assessment:")
                    print(json.dumps(
                        last_result,
                        indent=4,
                        ensure_ascii=False
                    ))

                except Exception as exc:
                    print("Analysis error:", exc)

                    last_result = {
                        "risk_level": "UNCERTAIN",
                        "status": "MANUAL_REVIEW",
                        "explanation": str(exc),
                        "recommendations": [
                            "Verify the camera feed and Gemini connection."
                        ]
                    }

                finally:
                    processing = False

            # Display camera feed.
            display = frame.copy()

            if last_result:
                risk = last_result.get(
                    "risk_level", "UNCERTAIN"
                )
                status = last_result.get(
                    "status", "UNKNOWN"
                )

                cv2.rectangle(
                    display,
                    (10, 10),
                    (650, 120),
                    (30, 30, 30),
                    -1
                )

                cv2.putText(
                    display,
                    f"RISK: {risk}",
                    (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if risk == "LOW" else (0, 165, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"STATUS: {status}",
                    (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            else:
                cv2.putText(
                    display,
                    "Waiting for first analysis...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                display,
                "Press Q to quit",
                (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(WINDOW_NAME, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()

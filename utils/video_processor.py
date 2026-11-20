
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2

from pipeline.video_graph import video_graph, reset_tracker


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_INTERVAL_SECONDS = 2.0


def process_source(source, source_name: str):

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open video source: {source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 120:
        fps = 20.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            "Could not read video dimensions."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{source_name}_{timestamp}"

    video_path = OUTPUT_DIR / f"{stem}_annotated.mp4"
    csv_path = OUTPUT_DIR / f"{stem}_detections.csv"
    json_path = OUTPUT_DIR / f"{stem}_summary.json"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        writer.release()

        raise RuntimeError(
            f"Could not create output video: {video_path}"
        )

    frame_number = 0
    processed_frames = 0
    review_frames = 0
    total_detections = 0
    peak_person_count = 0

    gemini_requests = 0
    gemini_failures = 0
    gemini_skipped_frames = 0

    latest_scene_analysis = {}
    latest_scene_error = ""
    latest_scene_status = "NOT_STARTED"
    latest_frame_quality = {}

    interrupted = False
    processing_error = None

    reset_tracker()

    gemini_interval_frames = max(
        1,
        int(round(fps * GEMINI_INTERVAL_SECONDS))
    )

    try:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            columns = [
                "frame_number",
                "time_seconds",
                "track_id",
                "class_name",
                "confidence",
                "bbox",
                "risk_level",
                "risk_explanation",
                "person_count",
                "scene_type",
                "scene_description",
                "hazard_count",
                "scene_available",
                "gemini_status",
                "gemini_skip_reason",
                "frame_mean_brightness",
                "frame_brightness_std",
            ]

            log = csv.DictWriter(
                file,
                fieldnames=columns,
            )

            log.writeheader()

            while True:

                ok, frame = capture.read()

                if not ok:
                    break

                frame_number += 1

                # Analyse the first frame and then sample
                # according to the configured interval.
                should_analyze_scene = (
                    frame_number == 1
                    or (frame_number - 1)
                    % gemini_interval_frames == 0
                )

                state = video_graph.invoke({

                    "frame": frame,
                    "frame_number": frame_number,

                    "detections": [],
                    "annotated_frame": frame,
                    "risk_result": {},

                    "analyze_scene": should_analyze_scene,
                    "scene_analysis": latest_scene_analysis,
                    "scene_error": latest_scene_error,
                    "scene_status": latest_scene_status,
                    "scene_skip_reason": "",
                    "frame_quality": latest_frame_quality,

                })

                detections = state["detections"]
                risk = state["risk_result"]
                annotated = state["annotated_frame"]

                # Retain the latest successful Gemini result.
                latest_scene_analysis = state.get(
                    "scene_analysis",
                    latest_scene_analysis
                )

                latest_scene_error = state.get(
                    "scene_error",
                    latest_scene_error
                )

                latest_scene_status = state.get(
                    "scene_status",
                    latest_scene_status
                )

                latest_frame_quality = state.get(
                    "frame_quality",
                    latest_frame_quality
                )

                # Count actual outcomes, not scheduled samples.
                current_scene_status = state.get(
                    "scene_status",
                    "NOT_SAMPLED"
                )

                if current_scene_status == "SUCCESS":
                    gemini_requests += 1

                elif current_scene_status == "FAILED":
                    gemini_requests += 1
                    gemini_failures += 1

                elif current_scene_status == "SKIPPED_LOW_INFORMATION":
                    gemini_skipped_frames += 1

                current_person_count = risk["person_count"]
                risk_level = risk["risk_level"]

                hazard_count = len(
                    risk.get("hazards_for_review", [])
                )

                peak_person_count = max(
                    peak_person_count,
                    current_person_count,
                )

                scene_type = risk.get(
                    "scene_type", "Unknown"
                )

                # ----------------------------------
                # VIDEO OVERLAY
                # ----------------------------------

                overlay_lines = [
                    f"Risk: {risk_level}",
                    f"People: {current_person_count}",
                    f"Peak: {peak_person_count}",
                    f"Scene: {scene_type}",
                    f"Hazards: {hazard_count}",
                ]

                if current_scene_status == "SKIPPED_LOW_INFORMATION":
                    overlay_lines.append("Gemini: Low information")

                elif current_scene_status == "FAILED":
                    overlay_lines.append("Gemini: Analysis failed")

                risk_color = (
                    (0, 0, 255)
                    if risk_level in (
                        "CROWD_REVIEW",
                        "HAZARD_REVIEW",
                        "MULTIPLE_REVIEW",
                    )
                    else (0, 200, 0)
                )

                for index, text in enumerate(overlay_lines):

                    cv2.putText(
                        annotated,
                        text,
                        (10, 30 + index * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        risk_color,
                        2,
                    )

                writer.write(annotated)

                processed_frames += 1
                total_detections += len(detections)

                if risk_level != "NO_RULE_TRIGGERED":
                    review_frames += 1

                # ----------------------------------
                # CSV LOGGING
                # ----------------------------------

                rows = detections if detections else [None]

                quality = latest_frame_quality or {}

                for item in rows:

                    log.writerow({

                        "frame_number": frame_number,

                        "time_seconds": round(
                            (frame_number - 1) / fps,
                            3,
                        ),

                        "track_id": (
                            item.get("track_id", "")
                            if item else ""
                        ),

                        "class_name": (
                            item.get("class_name", "")
                            if item else ""
                        ),

                        "confidence": (
                            round(item["confidence"], 4)
                            if item else ""
                        ),

                        "bbox": (
                            json.dumps(item["bbox"])
                            if item else ""
                        ),

                        "risk_level": risk_level,

                        "risk_explanation": risk["explanation"],

                        "person_count": current_person_count,

                        "scene_type": scene_type,

                        "scene_description": risk.get(
                            "scene_description", ""
                        ),

                        "hazard_count": hazard_count,

                        "scene_available": risk.get(
                            "scene_available", False
                        ),

                        "gemini_status": current_scene_status,

                        "gemini_skip_reason": (
                            state.get("scene_skip_reason", "")
                            if should_analyze_scene else ""
                        ),

                        "frame_mean_brightness": (
                            quality.get("mean_brightness", "")
                            if should_analyze_scene else ""
                        ),

                        "frame_brightness_std": (
                            quality.get("brightness_std", "")
                            if should_analyze_scene else ""
                        ),
                    })

                # ----------------------------------
                # DISPLAY
                # ----------------------------------

                cv2.imshow(
                    "Environmental Danger Prediction",
                    annotated,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Stopped by user using q.")
                    interrupted = True
                    break

    except KeyboardInterrupt:

        print(
            "\nCtrl+C received. Saving processed results."
        )

        interrupted = True

    except Exception as exc:

        processing_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"\nVideo processing failed: {processing_error}"
        )

    finally:

        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    # --------------------------------------
    # SUMMARY
    # --------------------------------------

    summary = {

        "source": source_name,

        "status": (
            "failed"
            if processing_error
            else "interrupted"
            if interrupted
            else "completed"
        ),

        "frames_processed": processed_frames,

        "detections_logged": total_detections,

        "frames_flagged_for_review": review_frames,

        "peak_person_count": peak_person_count,

        "gemini_model": "gemini-3.1-flash-lite",

        "gemini_sampling_interval_seconds":
            GEMINI_INTERVAL_SECONDS,

        "gemini_requests": gemini_requests,

        "gemini_failures": gemini_failures,

        "gemini_skipped_low_information_frames":
            gemini_skipped_frames,

        "latest_scene_status": latest_scene_status,

        "latest_frame_quality": latest_frame_quality,

        "latest_scene_analysis": latest_scene_analysis,

        "latest_gemini_error": latest_scene_error,

        "frame_quality_thresholds": {
            "black_frame_mean_brightness_max":
                8.0,
            "black_frame_brightness_std_max":
                5.0,
        },

        "risk_rules": [
            "CROWD_REVIEW when at least 5 people have confidence >= 0.50",
            "HAZARD_REVIEW when an observed Gemini hazard has confidence >= 0.70 and visual evidence",
            "MULTIPLE_REVIEW when both rules trigger",
        ],

        "confidence_threshold_for_detection": 0.35,

        "annotated_video": str(video_path),

        "detection_log": str(csv_path),

        "summary_file": str(json_path),
    }

    if processing_error:
        summary["error"] = processing_error

    try:

        json_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    except OSError as exc:

        print(f"Could not save JSON summary: {exc}")

    print("\nProcessing summary:")
    print(json.dumps(summary, indent=2))

    print(f"\nSaved video: {video_path}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved JSON:  {json_path}")

    return summary


from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from ultralytics import YOLO


class VideoState(TypedDict):
    frame: object
    frame_number: int
    detections: list[dict]
    annotated_frame: object
    risk_result: dict
    analyze_scene: bool
    scene_analysis: dict
    scene_error: str
    scene_status: str
    scene_skip_reason: str
    frame_quality: dict


model = YOLO("yolo11n.pt")

CONFIDENCE_THRESHOLD = 0.35


def reset_tracker() -> None:
    """Reset ByteTrack before processing an independent video."""

    predictor = getattr(model, "predictor", None)

    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", None)

    if not trackers:
        return

    for tracker in trackers:
        if hasattr(tracker, "reset"):
            tracker.reset()


def detection_node(state: VideoState) -> dict:
    """Detect and track objects in the current frame."""

    frame = state["frame"]

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            bbox = [
                int(value)
                for value in box.xyxy[0].tolist()
            ]

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0].item())

            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": confidence,
                "bbox": bbox,
                "track_id": track_id,
            })

    return {
        "detections": detections,
        "annotated_frame": result.plot(),
    }


def scene_understanding_node(state: VideoState) -> dict:
    """
    Run Gemini only on selected frames with sufficient
    visual information.
    """

    if not state.get("analyze_scene", False):
        return {
            "scene_status": "NOT_SAMPLED",
            "scene_error": "",
            "scene_skip_reason": "",
        }

    from agents.gemini_scene_agent import (
        analyze_scene,
        assess_frame_quality,
    )

    frame_number = state["frame_number"]

    try:

        quality = assess_frame_quality(state["frame"])

        if quality["is_low_information"]:

            reason = quality["reason"]

            print(
                f"Gemini skipped frame {frame_number}: {reason}"
            )

            return {
                "frame_quality": quality,
                "scene_status": "SKIPPED_LOW_INFORMATION",
                "scene_error": "",
                "scene_skip_reason": reason,
            }

        print(
            f"\nGemini analysing frame {frame_number}..."
        )

        scene_result = analyze_scene(state["frame"])

        print("Gemini scene analysis successful.")

        return {
            "scene_analysis": scene_result,
            "frame_quality": quality,
            "scene_status": "SUCCESS",
            "scene_error": "",
            "scene_skip_reason": "",
        }

    except Exception as error:

        error_message = (
            f"{type(error).__name__}: {error}"
        )

        print(
            f"Gemini analysis failed: {error_message}"
        )

        # Preserve the previous successful scene analysis.
        return {
            "scene_status": "FAILED",
            "scene_error": error_message,
            "scene_skip_reason": "",
        }


def risk_node(state: VideoState) -> dict:
    """Combine YOLO detections with the latest scene analysis."""

    from agents.risk_agent import analyze_risk

    risk_result = analyze_risk(
        detections=state["detections"],
        scene_analysis=state.get("scene_analysis")
    )

    return {
        "risk_result": risk_result,
    }


builder = StateGraph(VideoState)

builder.add_node(
    "detection_tracking",
    detection_node,
)

builder.add_node(
    "scene_understanding",
    scene_understanding_node,
)

builder.add_node(
    "risk_analysis",
    risk_node,
)

builder.add_edge(
    START,
    "detection_tracking",
)

builder.add_edge(
    "detection_tracking",
    "scene_understanding",
)

builder.add_edge(
    "scene_understanding",
    "risk_analysis",
)

builder.add_edge(
    "risk_analysis",
    END,
)

video_graph = builder.compile()

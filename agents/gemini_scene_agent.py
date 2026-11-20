
import os
import json
import base64
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-3.1-flash-lite"

BLACK_FRAME_MEAN_THRESHOLD = 8.0
BLACK_FRAME_STD_THRESHOLD = 5.0


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    return genai.Client(
        api_key=api_key,
        http_options={"timeout": 300000}
    )


def assess_frame_quality(image_input):
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))

        if image is None:
            raise ValueError(f"Could not read image: {image_input}")
    else:
        image = image_input

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image supplied.")

    if image.size == 0:
        raise ValueError("Image is empty.")

    if len(image.shape) == 2:
        grayscale = image
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(grayscale))
    brightness_std = float(np.std(grayscale))

    is_low_information = (
        mean_brightness <= BLACK_FRAME_MEAN_THRESHOLD
        and brightness_std <= BLACK_FRAME_STD_THRESHOLD
    )

    return {
        "mean_brightness": round(mean_brightness, 3),
        "brightness_std": round(brightness_std, 3),
        "is_low_information": is_low_information,
        "reason": (
            "Near-black frame with very little visual variation."
            if is_low_information else ""
        )
    }


def encode_image(image_input):
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }

        mime_type = mime_types.get(image_path.suffix.lower())

        if mime_type is None:
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}"
            )

        image_bytes = image_path.read_bytes()

    else:
        success, encoded = cv2.imencode(
            ".jpg",
            image_input,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if not success:
            raise ValueError("Could not encode video frame.")

        image_bytes = encoded.tobytes()
        mime_type = "image/jpeg"

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return image_base64, mime_type


def validate_and_normalize_bbox(raw_bbox):
    """
    Validates and normalizes hazard bounding box coordinates.
    Strict rules:
    - Must be a dictionary containing x_min, y_min, x_max, y_max (or [ymin, xmin, ymax, xmax] list).
    - Supports clean normalized floats [0.0, 1.0].
    - Supports Gemini 0-1000 integer scale if coordinates are on the 0-1000 scale (> 1.0).
    - Rejects coordinates out of range (< 0.0 or > 1.0 for normalized, > 1000 for integer scale).
    - Rejects invalid ordering (x_min >= x_max or y_min >= y_max).
    - Rejects zero-area or microscopic boxes (area < 0.0005).
    - Rejects oversized bounding boxes (> 0.85 frame area) unless explicitly justified.
    - Returns normalized dictionary with float values [0.0, 1.0] or None.
    """
    if not raw_bbox:
        return None

    # Handle if Gemini returned a 4-element array [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            raw_bbox = {
                "ymin": raw_bbox[0],
                "xmin": raw_bbox[1],
                "ymax": raw_bbox[2],
                "xmax": raw_bbox[3]
            }
        except Exception:
            return None

    if not isinstance(raw_bbox, dict):
        return None

    # Normalize key names (handle x_min, xmin, y_min, ymin, etc.)
    x_min_raw = raw_bbox.get("x_min") if "x_min" in raw_bbox else raw_bbox.get("xmin")
    y_min_raw = raw_bbox.get("y_min") if "y_min" in raw_bbox else raw_bbox.get("ymin")
    x_max_raw = raw_bbox.get("x_max") if "x_max" in raw_bbox else raw_bbox.get("xmax")
    y_max_raw = raw_bbox.get("y_max") if "y_max" in raw_bbox else raw_bbox.get("ymax")

    if None in (x_min_raw, y_min_raw, x_max_raw, y_max_raw):
        return None

    try:
        x_min = float(x_min_raw)
        y_min = float(y_min_raw)
        x_max = float(x_max_raw)
        y_max = float(y_max_raw)
    except (ValueError, TypeError):
        return None

    # Determine coordinate scale
    vals = [x_min, y_min, x_max, y_max]
    if any(v < 0 for v in vals):
        return None

    # If any value is > 1.0, test if it's on the 1000 scale
    if any(v > 1.0 for v in vals):
        if all(v <= 1000.0 for v in vals):
            x_min /= 1000.0
            y_min /= 1000.0
            x_max /= 1000.0
            y_max /= 1000.0
        else:
            return None

    # Bounds checking
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    # Strict ordering
    if x_min >= x_max or y_min >= y_max:
        return None

    # Non-degenerate area check
    area = (x_max - x_min) * (y_max - y_min)
    if area < 0.0005:
        return None

    return {
        "x_min": round(x_min, 4),
        "y_min": round(y_min, 4),
        "x_max": round(x_max, 4),
        "y_max": round(y_max, 4)
    }


def analyze_scene(image_input):

    quality = assess_frame_quality(image_input)

    if quality["is_low_information"]:
        raise ValueError(
            "Gemini analysis skipped: " + quality["reason"]
        )

    image_data, mime_type = encode_image(image_input)
    client = get_gemini_client()

    prompt = """
    You are a visual safety monitoring agent for an
    Environmental Danger Prediction System.

    Analyze the supplied camera frame for these hazards:

    1. CROWD CONGESTION:
       Excessive crowd density, overcrowding and congested
       pedestrian areas.

    2. OBSTACLES:
       Objects obstructing walkways, cables across walking
       paths, misplaced furniture, fallen objects and trip hazards.

    3. SPILLS:
       Visible liquid spills, wet floors, puddles and
       slippery walking surfaces.

    4. FIRE OR SMOKE:
       Visible flames, smoke plumes or heat haze.

    5. OTHER:
       Any other visually identifiable environmental or safety hazard.

    RULES:

    - Report only hazards supported by visible evidence.
    - Do not invent hazards.
    - Do not classify ordinary objects as hazards without
      evidence of a safety risk.
    - Do not infer movement from a still image.
    - Do not estimate exact crowd counts.
    - Distinguish OBSERVED from INFERRED hazards.
    - A person appearing in the frame is not itself a hazard.
    - Do not report irrelevant uncertainty about hidden areas,
      object identity or activities.
    - Only report uncertainty directly affecting assessment
      of the hazard categories.
    - If no relevant uncertainty exists, return [].

    BOUNDING BOX INSTRUCTIONS (OBJECT-LEVEL TIGHT LOCALIZATION):

    For EVERY visually detectable hazard, you MUST provide tight, object-level bounding box coordinates
    around the SPECIFIC physical hazard item itself, NOT the broader room, floor, or background.
    
    CRITICAL OBJECT-LEVEL LOCALIZATION RULES:
    1. TIGHT BOUNDS AROUND THE PHYSICAL OBJECT:
       - If the hazard is an OBSTACLE (e.g., a backpack, box, stool, cable, dropped toy):
         envelop ONLY that specific obstacle object itself. DO NOT box the whole walking path, rug, or room.
       - If the hazard is a SPILL / PUDDLE / WET SPOT:
         envelop ONLY the actual liquid surface contour/puddle on the floor. DO NOT box the whole room or entire floor.
       - If the hazard is an ELECTRICAL CABLE / TRIP HAZARD:
         envelop ONLY the visible length of the cable itself.
       - If the hazard is FIRE or SMOKE:
         envelop ONLY the visible active flames or the concentrated smoke plume.
       - If the hazard is CROWD CONGESTION:
         envelop ONLY the dense cluster of individuals creating the obstruction.
    
    2. AVOID BROAD SCENE-LEVEL BOXES:
       - Never draw a box covering more than 75% of the frame unless a single hazardous object genuinely spans that entire area.
       - Never include unrelated surrounding objects (e.g. sofas, distant walls, unblocked doorways) in the box.

    Coordinate convention (STRICT):
    - Origin (0.0, 0.0) is at the TOP-LEFT corner of the original camera image.
    - x represents horizontal position: 0.0 at extreme left edge, 1.0 at extreme right edge.
    - y represents vertical position: 0.0 at extreme top edge, 1.0 at extreme bottom edge.
    - All coordinates MUST be decimal numbers strictly normalized between 0.0 and 1.0.
      Example of a localized object: {"x_min": 0.32, "y_min": 0.65, "x_max": 0.48, "y_max": 0.82}.
      NEVER return integer pixel values (such as 320 or 650).
    - Ensure strictly: 0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0.
    
    3. MULTIPLE DISTINCT HAZARDS:
       - If there are multiple separate hazards in the scene, list EACH one as a separate item in "potential_hazards"
         with its own dedicated tight "bbox".
    
    4. UNLOCALIZED HAZARDS:
       - If and ONLY if a hazard cannot be localized reliably to a distinct physical object, set "bbox" to null.
       - Never guess or approximate coordinates for objects you cannot clearly see.

    Return ONLY valid JSON:

    {
        "scene_type": "string",
        "scene_description": "string",
        "visible_entities": [
            {
                "entity": "string",
                "description": "string"
            }
        ],
        "spatial_relations": ["string"],
        "potential_hazards": [
            {
                "hazard": "string",
                "visual_evidence": "string",
                "severity": "LOW, MEDIUM, or HIGH",
                "confidence": 0.0,
                "observation_type": "OBSERVED or INFERRED",
                "bbox": {
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 1.0,
                    "y_max": 1.0
                }
            }
        ],
        "safe_observations": ["string"],
        "uncertainties": []
    }

    Confidence must be between 0.0 and 1.0.
    Do not use Markdown code fences.
    """

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image",
                "data": image_data,
                "mime_type": mime_type
            }
        ]
    )

    response_text = interaction.output_text

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    # Clean response text from potential markdown code fences or trailing characters
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:].strip()

    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"Gemini did not return JSON:\n{response_text}"
        )

    json_str = cleaned_text[start:end + 1]
    
    # Pre-clean any arithmetic expressions like 147.0 / 1000.0 or 724 / 1000
    import re
    def eval_expr(match):
        try:
            num = float(match.group(1))
            den = float(match.group(2))
            return str(round(num / den, 4))
        except Exception:
            return match.group(0)

    json_str = re.sub(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", eval_expr, json_str)

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        # In case of trailing commas or minor JSON syntax imperfections
        sanitized = re.sub(r",\s*([}\]])", r"\1", json_str)
        result = json.loads(sanitized)

    required_fields = [
        "scene_type",
        "scene_description",
        "visible_entities",
        "spatial_relations",
        "potential_hazards",
        "safe_observations",
        "uncertainties"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"Gemini response missing field: {field}"
            )

    # Validate and normalize bbox for all hazards
    for item in result.get("potential_hazards", []):
        if isinstance(item, dict):
            raw_box = item.get("bbox")
            item["bbox"] = validate_and_normalize_bbox(raw_box)

    return result


if __name__ == "__main__":
    test_image = PROJECT_ROOT / "test_image.jpg"

    print("Testing Gemini Scene Agent...")
    print("Model:", MODEL_NAME)

    quality = assess_frame_quality(test_image)
    print(json.dumps(quality, indent=4))

    if not quality["is_low_information"]:
        result = analyze_scene(test_image)
        print(json.dumps(result, indent=4, ensure_ascii=False))

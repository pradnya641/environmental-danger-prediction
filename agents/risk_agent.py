
def analyze_risk(scene_analysis: dict) -> dict:
    """Assess environmental risks using Gemini observations."""

    if not isinstance(scene_analysis, dict):
        raise ValueError("scene_analysis must be a dictionary")

    hazards = scene_analysis.get("potential_hazards", [])
    uncertainties = scene_analysis.get("uncertainties", [])

    if not isinstance(hazards, list):
        hazards = []

    if not isinstance(uncertainties, list):
        uncertainties = []

    categories = {
        "CROWD_CONGESTION": [
            "crowd", "crowding", "congestion",
            "overcrowd", "dense gathering"
        ],
        "OBSTACLE": [
            "obstacle", "obstruction", "blocked walkway",
            "blocked path", "trip hazard", "blockage"
        ],
        "SPILL": [
            "spill", "liquid on floor", "wet floor",
            "leak", "slippery surface"
        ]
    }

    reviewed_hazards = []
    uncertain_hazards = []

    for item in hazards:
        if not isinstance(item, dict):
            continue

        name = str(item.get("hazard", "")).strip()
        evidence = str(item.get("visual_evidence", "")).strip()
        observation = str(
            item.get("observation_type", "")
        ).upper()

        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        if not name:
            continue

        search_text = f"{name} {evidence}".lower()

        category = "OTHER"

        for category_name, keywords in categories.items():
            if any(keyword in search_text for keyword in keywords):
                category = category_name
                break

        result = {
            "hazard": name,
            "category": category,
            "visual_evidence": evidence,
            "severity": str(
                item.get("severity", "UNKNOWN")
            ).upper(),
            "confidence": confidence,
            "observation_type": observation
        }

        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            result["bbox"] = bbox

        if observation == "OBSERVED" and evidence and confidence >= 0.70:
            reviewed_hazards.append(result)
        else:
            uncertain_hazards.append(result)

    severity_order = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3
    }

    highest_severity = max(
        (
            severity_order.get(h["severity"], 0)
            for h in reviewed_hazards
        ),
        default=0
    )

    if reviewed_hazards:
        risk_level = {
            3: "HIGH",
            2: "MEDIUM",
            1: "LOW"
        }.get(highest_severity, "UNCERTAIN")

        status = "REVIEW_REQUIRED"

    elif uncertain_hazards or uncertainties:
        risk_level = "UNCERTAIN"
        status = "MANUAL_REVIEW"

    else:
        risk_level = "LOW"
        status = "NO_RULE_TRIGGERED"

    recommendations = {
        "CROWD_CONGESTION":
            "Review crowd density and consider managing pedestrian flow.",
        "OBSTACLE":
            "Inspect the obstruction and assess whether it blocks a walkway.",
        "SPILL":
            "Inspect the spill and consider restricting access until it is addressed.",
        "OTHER":
            "Manually inspect the reported potential hazard."
    }

    actions = list(dict.fromkeys(
        recommendations[h["category"]]
        for h in reviewed_hazards
    ))

    if uncertain_hazards or uncertainties:
        actions.append(
            "Verify uncertain visual observations before taking corrective action."
        )

    if not actions:
        actions.append(
            "Continue monitoring. No configured hazard rule was triggered."
        )

    return {
        "risk_level": risk_level,
        "status": status,
        "explanation": (
            f"{len(reviewed_hazards)} supported hazard(s), "
            f"{len(uncertain_hazards)} uncertain hazard(s), "
            f"and {len(uncertainties)} scene uncertainty/uncertainties."
        ),
        "hazards_for_review": reviewed_hazards,
        "uncertain_hazards": uncertain_hazards,
        "scene_uncertainties": uncertainties,
        "recommendations": actions,
        "scene_type": scene_analysis.get("scene_type", "Unknown"),
        "scene_description": scene_analysis.get("scene_description", "")
    }

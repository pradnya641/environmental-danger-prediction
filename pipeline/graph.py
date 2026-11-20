
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from agents.gemini_scene_agent import analyze_scene
from agents.risk_agent import analyze_risk


class DangerState(TypedDict):
    image_input: Any
    scene_analysis: dict
    risk_result: dict
    error: str


def scene_understanding_node(state: DangerState) -> dict:
    """Analyze the supplied image or OpenCV frame using Gemini."""

    try:
        scene = analyze_scene(state["image_input"])
        return {
            "scene_analysis": scene,
            "error": ""
        }

    except Exception as exc:
        return {
            "scene_analysis": {},
            "error": str(exc)
        }


def risk_assessment_node(state: DangerState) -> dict:
    """Assess risks from Gemini's structured scene analysis."""

    if state.get("error") or not state.get("scene_analysis"):
        return {
            "risk_result": {
                "risk_level": "UNCERTAIN",
                "status": "MANUAL_REVIEW",
                "explanation": (
                    "Scene analysis was unavailable. "
                    "The environment cannot be classified."
                ),
                "hazards_for_review": [],
                "uncertain_hazards": [],
                "recommendations": [
                    "Check camera visibility and Gemini connectivity."
                ]
            }
        }

    result = analyze_risk(state["scene_analysis"])

    return {"risk_result": result}


builder = StateGraph(DangerState)

builder.add_node("scene_understanding", scene_understanding_node)
builder.add_node("risk_assessment", risk_assessment_node)

builder.add_edge(START, "scene_understanding")
builder.add_edge("scene_understanding", "risk_assessment")
builder.add_edge("risk_assessment", END)

danger_graph = builder.compile()


def analyze_image(image_input: Any) -> dict:
    """Run the complete LangGraph pipeline on an image or frame."""

    result = danger_graph.invoke({
        "image_input": image_input,
        "scene_analysis": {},
        "risk_result": {},
        "error": ""
    })

    return {
        "scene_analysis": result["scene_analysis"],
        "risk_result": result["risk_result"],
        "error": result["error"]
    }


if __name__ == "__main__":
    from pathlib import Path

    image_path = Path(__file__).resolve().parent.parent / "test_image.jpg"

    print("Running Environmental Danger Prediction System...")

    result = analyze_image(str(image_path))

    print("\nSCENE ANALYSIS")
    print(result["scene_analysis"])

    print("\nRISK ASSESSMENT")
    print(result["risk_result"])

    if result["error"]:
        print("\nERROR:", result["error"])

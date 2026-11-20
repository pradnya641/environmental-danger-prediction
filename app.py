
from datetime import datetime, timezone
from typing import Literal

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from pipeline.graph import analyze_image
from database import SessionLocal
from models import Incident, Hazard


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Environmental Danger Prediction System",
    description="AI-powered environmental hazard monitoring",
    version="1.0.0"
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class IncidentStatusUpdate(BaseModel):
    status: Literal["ACKNOWLEDGED", "RESOLVED"]


# --------------------------------------------------
# RESPONSE HELPERS
# --------------------------------------------------

def serialize_hazard(hazard):
    return {
        "hazard_id": hazard.hazard_id,
        "hazard_type": hazard.hazard_type,
        "category": hazard.category,
        "visual_evidence": hazard.visual_evidence,
        "severity": hazard.severity,
        "confidence": (
            float(hazard.confidence)
            if hazard.confidence is not None else None
        ),
        "observation_type": hazard.observation_type
    }


def serialize_incident(incident, include_hazards=False):
    data = {
        "incident_id": incident.incident_id,
        "camera_id": incident.camera_id,
        "scene_type": incident.scene_type,
        "scene_description": incident.scene_description,
        "risk_level": incident.risk_level,
        "status": incident.status,
        "analyzed_at": (
            incident.analyzed_at.isoformat()
            if incident.analyzed_at else None
        )
    }

    if include_hazards:
        data["hazards"] = [
            serialize_hazard(hazard)
            for hazard in incident.hazards
        ]

    return data


# --------------------------------------------------
# DATABASE STORAGE
# --------------------------------------------------

def save_analysis_to_database(result: dict) -> int:
    db = SessionLocal()

    try:
        scene = result["scene_analysis"] or {}
        risk = result["risk_result"] or {}

        valid_risk_levels = {
            "LOW", "MEDIUM", "HIGH", "UNCERTAIN"
        }

        valid_statuses = {
            "NO_RULE_TRIGGERED",
            "REVIEW_REQUIRED",
            "MANUAL_REVIEW",
            "ACKNOWLEDGED",
            "RESOLVED"
        }

        risk_level = str(
            risk.get("risk_level", "UNCERTAIN")
        ).upper()

        status = str(
            risk.get("status", "NO_RULE_TRIGGERED")
        ).upper()

        if risk_level not in valid_risk_levels:
            risk_level = "UNCERTAIN"

        if status not in valid_statuses:
            status = "MANUAL_REVIEW"

        incident = Incident(
            scene_type=scene.get("scene_type"),
            scene_description=scene.get("scene_description"),
            risk_level=risk_level,
            status=status
        )

        db.add(incident)
        db.flush()

        hazards_to_save = (
            risk.get("hazards_for_review", [])
            + risk.get("uncertain_hazards", [])
        )

        for item in hazards_to_save:
            severity = str(
                item.get("severity", "UNCERTAIN")
            ).upper()

            if severity not in valid_risk_levels:
                severity = "UNCERTAIN"

            confidence = item.get("confidence")

            if confidence is not None:
                confidence = max(
                    0.0,
                    min(1.0, float(confidence))
                )

            hazard = Hazard(
                incident_id=incident.incident_id,
                hazard_type=(
                    item.get("hazard")
                    or item.get("hazard_type")
                    or "UNKNOWN"
                ),
                category=item.get("category"),
                visual_evidence=item.get("visual_evidence"),
                severity=severity,
                confidence=confidence,
                observation_type=item.get("observation_type")
            )

            db.add(hazard)

        db.commit()

        return incident.incident_id

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# --------------------------------------------------
# GENERAL ENDPOINTS
# --------------------------------------------------

@app.get("/", tags=["General"])
def root():
    return {
        "application": "Environmental Danger Prediction System",
        "message": "Backend is running.",
        "documentation": "/docs"
    }


@app.get("/health", tags=["General"])
def health():
    return {
        "status": "healthy",
        "application": "Environmental Danger Prediction System",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

@app.post(
    "/api/analyze/image",
    tags=["Hazard Analysis"]
)
async def analyze_uploaded_image(
    file: UploadFile = File(...)
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only JPG, PNG and WEBP images are supported."
        )

    try:
        image_bytes = await file.read(
            10 * 1024 * 1024 + 1
        )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB."
            )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable image."
            )

        result = analyze_image(image)

        if result.get("error"):
            raise HTTPException(
                status_code=502,
                detail="Image analysis failed."
            )

        try:
            incident_id = save_analysis_to_database(result)

        except Exception as exc:
            print(f"Database storage error: {exc}")
            raise HTTPException(
                status_code=500,
                detail="Analysis completed, but saving to MySQL failed."
            )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "incident_id": incident_id,
            "scene_analysis": result["scene_analysis"],
            "risk_result": result["risk_result"],
            "error": None
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Image analysis error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Image analysis failed."
        )

    finally:
        await file.close()


# --------------------------------------------------
# 1. GET INCIDENT HISTORY
# --------------------------------------------------

@app.get(
    "/api/incidents",
    tags=["Incident Management"]
)
def get_incidents(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    risk_level: str | None = Query(None),
    status: str | None = Query(None)
):
    db = SessionLocal()

    try:
        query = db.query(Incident)

        if risk_level:
            allowed = {"LOW", "MEDIUM", "HIGH", "UNCERTAIN"}
            risk_level = risk_level.upper()

            if risk_level not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid risk_level."
                )

            query = query.filter(
                Incident.risk_level == risk_level
            )

        if status:
            allowed = {
                "NO_RULE_TRIGGERED",
                "REVIEW_REQUIRED",
                "MANUAL_REVIEW",
                "ACKNOWLEDGED",
                "RESOLVED"
            }

            status = status.upper()

            if status not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid incident status."
                )

            query = query.filter(
                Incident.status == status
            )

        total = query.count()

        records = (
            query.order_by(Incident.analyzed_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "incidents": [
                serialize_incident(record)
                for record in records
            ]
        }

    finally:
        db.close()


# --------------------------------------------------
# 2. GET INCIDENT DETAILS
# --------------------------------------------------

@app.get(
    "/api/incidents/{incident_id}",
    tags=["Incident Management"]
)
def get_incident_details(incident_id: int):
    db = SessionLocal()

    try:
        incident = db.query(Incident).filter(
            Incident.incident_id == incident_id
        ).first()

        if incident is None:
            raise HTTPException(
                status_code=404,
                detail="Incident not found."
            )

        return serialize_incident(
            incident,
            include_hazards=True
        )

    finally:
        db.close()


# --------------------------------------------------
# 3. DASHBOARD STATISTICS
# --------------------------------------------------

@app.get(
    "/api/dashboard/stats",
    tags=["Dashboard"]
)
def get_dashboard_stats():
    db = SessionLocal()

    try:
        total_incidents = db.query(
            func.count(Incident.incident_id)
        ).scalar() or 0

        total_hazards = db.query(
            func.count(Hazard.hazard_id)
        ).scalar() or 0

        risk_distribution = {
            level: count
            for level, count in db.query(
                Incident.risk_level,
                func.count(Incident.incident_id)
            ).group_by(Incident.risk_level).all()
        }

        status_distribution = {
            status: count
            for status, count in db.query(
                Incident.status,
                func.count(Incident.incident_id)
            ).group_by(Incident.status).all()
        }

        hazard_distribution = {
            hazard_type: count
            for hazard_type, count in db.query(
                Hazard.hazard_type,
                func.count(Hazard.hazard_id)
            ).group_by(Hazard.hazard_type).all()
        }

        high_risk_incidents = db.query(
            func.count(Incident.incident_id)
        ).filter(
            Incident.risk_level == "HIGH"
        ).scalar() or 0

        review_required = db.query(
            func.count(Incident.incident_id)
        ).filter(
            Incident.status.in_([
                "REVIEW_REQUIRED",
                "MANUAL_REVIEW"
            ])
        ).scalar() or 0

        recent_records = (
            db.query(Incident)
            .order_by(Incident.analyzed_at.desc())
            .limit(5)
            .all()
        )

        return {
            "total_incidents": total_incidents,
            "total_hazards": total_hazards,
            "high_risk_incidents": high_risk_incidents,
            "review_required": review_required,
            "risk_distribution": risk_distribution,
            "status_distribution": status_distribution,
            "hazard_distribution": hazard_distribution,
            "recent_incidents": [
                serialize_incident(record)
                for record in recent_records
            ]
        }

    finally:
        db.close()


# --------------------------------------------------
# 4. ACKNOWLEDGE / RESOLVE INCIDENT
# --------------------------------------------------

@app.patch(
    "/api/incidents/{incident_id}/status",
    tags=["Incident Management"]
)
def update_incident_status(
    incident_id: int,
    update: IncidentStatusUpdate
):
    db = SessionLocal()

    try:
        incident = db.query(Incident).filter(
            Incident.incident_id == incident_id
        ).first()

        if incident is None:
            raise HTTPException(
                status_code=404,
                detail="Incident not found."
            )

        if incident.status == "RESOLVED":
            raise HTTPException(
                status_code=409,
                detail="Resolved incidents cannot be changed."
            )

        incident.status = update.status

        db.commit()
        db.refresh(incident)

        return {
            "message": "Incident status updated successfully.",
            "incident": serialize_incident(incident)
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# --------------------------------------------------
# 5. NON-PERSISTENT FRAME ANALYSIS (live webcam)
# --------------------------------------------------

@app.post(
    "/api/analyze/frame",
    tags=["Hazard Analysis"]
)
async def analyze_live_frame(
    file: UploadFile = File(...)
):
    """
    Analyse a single webcam frame with Gemini Vision + LangGraph
    and return the result WITHOUT writing to the database.

    The React frontend calls this endpoint every N seconds during
    live monitoring.  Persistence is handled separately by the
    frontend, which calls /api/analyze/image on a cooldown when a
    significant hazard is confirmed.
    """
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only JPG, PNG and WEBP images are supported."
        )

    try:
        image_bytes = await file.read(10 * 1024 * 1024 + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded frame is empty."
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Frame size must not exceed 10 MB."
            )

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable image frame."
            )

        result = analyze_image(image)

        if result.get("error"):
            # Provide structured diagnostic info rather than an opaque 502 error
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scene_analysis": result.get("scene_analysis") or {
                    "scene_type": "Camera Feed (Analysis Pending)",
                    "scene_description": result.get("error", "Waiting for clear camera frame..."),
                    "potential_hazards": []
                },
                "risk_result": result.get("risk_result") or {
                    "risk_level": "UNCERTAIN",
                    "status": "MANUAL_REVIEW",
                    "explanation": result.get("error", "Analyzing frame..."),
                    "hazards_for_review": [],
                    "uncertain_hazards": [],
                    "recommendations": ["Ensure camera lens is unobstructed and well-lit."]
                },
                "error": result["error"]
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scene_analysis": result["scene_analysis"],
            "risk_result": result["risk_result"],
            "error": None
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Live frame analysis error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Live frame analysis failed."
        )

    finally:
        await file.close()

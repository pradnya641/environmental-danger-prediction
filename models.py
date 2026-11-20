
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    Enum,
    DateTime,
    DECIMAL,
    ForeignKey,
    text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    role = Column(
        Enum("ADMIN", "OPERATOR", "VIEWER"),
        server_default=text("'VIEWER'")
    )

    created_at = Column(
        DateTime,
        server_default=func.current_timestamp()
    )


class Camera(Base):
    __tablename__ = "cameras"

    camera_id = Column(Integer, primary_key=True, autoincrement=True)
    camera_name = Column(String(100), nullable=False)
    location = Column(String(255))

    source_type = Column(
        Enum("WEBCAM", "IP_CAMERA", "VIDEO"),
        server_default=text("'WEBCAM'")
    )

    status = Column(
        Enum("ACTIVE", "INACTIVE"),
        server_default=text("'INACTIVE'")
    )

    created_at = Column(
        DateTime,
        server_default=func.current_timestamp()
    )

    incidents = relationship(
        "Incident",
        back_populates="camera"
    )


class Incident(Base):
    __tablename__ = "incidents"

    incident_id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    camera_id = Column(
        Integer,
        ForeignKey("cameras.camera_id", ondelete="SET NULL"),
        nullable=True
    )

    scene_type = Column(String(150))
    scene_description = Column(Text)

    risk_level = Column(
        Enum("LOW", "MEDIUM", "HIGH", "UNCERTAIN"),
        server_default=text("'UNCERTAIN'")
    )

    status = Column(
        Enum(
            "NO_RULE_TRIGGERED",
            "REVIEW_REQUIRED",
            "MANUAL_REVIEW",
            "ACKNOWLEDGED",
            "RESOLVED"
        ),
        server_default=text("'NO_RULE_TRIGGERED'")
    )

    analyzed_at = Column(
        DateTime,
        server_default=func.current_timestamp()
    )

    camera = relationship(
        "Camera",
        back_populates="incidents"
    )

    hazards = relationship(
        "Hazard",
        back_populates="incident",
        cascade="all, delete-orphan"
    )


class Hazard(Base):
    __tablename__ = "hazards"

    hazard_id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    incident_id = Column(
        BigInteger,
        ForeignKey("incidents.incident_id", ondelete="CASCADE"),
        nullable=False
    )

    hazard_type = Column(String(100), nullable=False)
    category = Column(String(100))
    visual_evidence = Column(Text)

    severity = Column(
        Enum("LOW", "MEDIUM", "HIGH", "UNCERTAIN"),
        server_default=text("'UNCERTAIN'")
    )

    confidence = Column(DECIMAL(4, 3))
    observation_type = Column(String(30))

    incident = relationship(
        "Incident",
        back_populates="hazards"
    )

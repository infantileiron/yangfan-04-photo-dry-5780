from datetime import datetime

from sqlalchemy import JSON, Boolean, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, new_id, utcnow


class ModelState(Base):
    __tablename__ = "model_state"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default="active")
    active_key: Mapped[str] = mapped_column(String(64), default="cross-ridge-v1", index=True)
    active_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(160), nullable=True)
    adoption_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class ModelValidation(Base):
    __tablename__ = "model_validations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    training_dataset_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    label: Mapped[str] = mapped_column(String(100), default="留验比较")
    parameter: Mapped[float | None] = mapped_column(Float, nullable=True)
    base_model_key: Mapped[str] = mapped_column(String(64), index=True)
    base_fingerprint: Mapped[str] = mapped_column(String(64))
    base_state_adoption_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    candidate_model_key: Mapped[str] = mapped_column(String(64), index=True)
    candidate_fingerprint: Mapped[str] = mapped_column(String(64))
    eligible: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    report: Mapped[dict] = mapped_column(JSON)
    actor: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


class ModelAdoption(Base):
    __tablename__ = "model_adoptions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    validation_id: Mapped[str] = mapped_column(ForeignKey("model_validations.id"), unique=True, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(120), unique=True, nullable=True, index=True)
    base_key: Mapped[str] = mapped_column(String(64))
    candidate_key: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="applied")
    actor: Mapped[str] = mapped_column(String(160))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    applied_at: Mapped[datetime | None] = mapped_column(nullable=True)


# Descriptive aliases for code/tests using “switch” terminology.
ActiveModel = ModelState
ModelSwitch = ModelAdoption
HoldoutValidation = ModelValidation


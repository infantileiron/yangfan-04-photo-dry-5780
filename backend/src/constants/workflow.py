from enum import Enum


class Role(str, Enum):
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class Stage(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Decision(str, Enum):
    APPROVE = "approve"
    HOLD = "hold"


class ModelStatus(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    ARCHIVED = "archived"


class ValidationStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    STALE = "stale"


class AdoptionStatus(str, Enum):
    APPLIED = "applied"
    FAILED = "failed"
    PENDING = "pending"

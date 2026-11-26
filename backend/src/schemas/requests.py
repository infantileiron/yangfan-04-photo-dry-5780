from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints
from typing import Annotated

from src.constants.workflow import Decision

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Request):
    email: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    password: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class Import(Request):
    name: Name
    csv: Annotated[str, StringConstraints(min_length=1, max_length=1_000_000)]


class RunRequest(Request):
    dataset_id: Annotated[str, StringConstraints(min_length=36, max_length=36)]
    label: Name
    parameter: FiniteFloat
    model_key: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] = None


class ValidationRequest(Request):
    model_config = ConfigDict(extra="allow")

    dataset_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    holdout_dataset_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    training_dataset_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    history_dataset_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    base_dataset_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    baseline_dataset_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    label: Name = "留验比较"
    parameter: FiniteFloat | None = None
    coverage_pct: FiniteFloat | None = None
    base_model_key: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] = None
    old_model_key: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] = None
    candidate_model_key: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] = None
    new_model_key: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] = None
    base_parameter: FiniteFloat | None = None
    candidate_parameter: FiniteFloat | None = None
    holdout_dates: list[Annotated[str, StringConstraints(min_length=8, max_length=10)]] | None = None
    idempotency_key: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)] = None


class AdoptionRequest(Request):
    model_config = ConfigDict(extra="allow")

    validation_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    comparison_id: Annotated[str | None, StringConstraints(min_length=1, max_length=36)] = None
    idempotency_key: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)] = None
    request_id: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)] = None
    expected_active_model: Annotated[str | None, StringConstraints(min_length=1, max_length=64)] = None


class ReviewRequest(Request):
    decision: Decision
    samples: Annotated[list[Annotated[str, StringConstraints(min_length=1, max_length=120)]], Field(max_length=500)]
    note: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=1000)]

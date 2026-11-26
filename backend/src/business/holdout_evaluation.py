"""Train old/candidate recipes and compare them on isolated holdout strips."""

from __future__ import annotations

from typing import Any, Iterable

from .model_adoption import DEFAULT_CONDITIONS, compare_holdout
from .model_recipes import DEFAULT_MODEL_KEY, fit_recipe, public_recipe, resolve_parameter
from .photo_recipe import FEATURES, parse_bundle, parse_holdout_rows


def _trial_record(trial, old_seconds: int, new_seconds: int) -> dict[str, Any]:
    record = {
        "sample_id": trial.sample_id,
        "trial_date": trial.day.isoformat(),
        "old_seconds": old_seconds,
        "new_seconds": new_seconds,
        "measured_s": trial.measured_s,
    }
    record.update({field: value for field, value in zip(FEATURES, trial.features)})
    return record


def evaluate_rows(
    rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]] | None = None,
    *,
    base_model_key: str | None = None,
    candidate_model_key: str | None = None,
    parameter: float | None = None,
    base_parameter: float | None = None,
    candidate_parameter: float | None = None,
    holdout_dates: Iterable[str] | None = None,
    conditions=DEFAULT_CONDITIONS,
    confidence: float = 0.95,
    min_dates: int = 2,
) -> dict[str, Any]:
    base_key = base_model_key or DEFAULT_MODEL_KEY
    candidate_key = candidate_model_key or "main-ridge-v1"
    # Validate keys before parsing so callers get a clear model error.
    base_info = public_recipe(base_key)
    candidate_info = public_recipe(candidate_key)

    if holdout_rows is not None:
        separate_holdout = parse_holdout_rows(holdout_rows)
        separate_dates = {trial.day.isoformat() for trial in separate_holdout}
        all_holdout_dates = set(holdout_dates or ()) | separate_dates
    else:
        all_holdout_dates = set(holdout_dates or ())
    history, _candidates, embedded_holdout = parse_bundle(
        rows, holdout_dates=all_holdout_dates, require_candidates=False,
    )
    holdout = separate_holdout if holdout_rows is not None else embedded_holdout
    base_coverage = resolve_parameter(base_key, base_parameter if base_parameter is not None else parameter)
    candidate_coverage = resolve_parameter(
        candidate_key, candidate_parameter if candidate_parameter is not None else parameter
    )
    base_fitted = fit_recipe(history, base_coverage, base_key)
    candidate_fitted = fit_recipe(history, candidate_coverage, candidate_key)

    records = [
        _trial_record(
            trial,
            base_fitted.recommended_seconds(trial),
            candidate_fitted.recommended_seconds(trial),
        )
        for trial in holdout
    ]
    report = compare_holdout(
        records, conditions=conditions, confidence=confidence, min_dates=min_dates,
    )
    holdout_days = sorted({trial.day.isoformat() for trial in holdout})
    report.update({
        "base_model": {
            **base_info,
            "parameter": base_coverage,
            "margin_s": round(base_fitted.margin, 4),
            "interactions_used": list(base_fitted.interactions),
        },
        "candidate_model": {
            **candidate_info,
            "parameter": candidate_coverage,
            "margin_s": round(candidate_fitted.margin, 4),
            "interactions_used": list(candidate_fitted.interactions),
        },
    })
    report["old_model"] = report["base_model"]
    report["new_model"] = report["candidate_model"]
    report["dataset"] = {
        "holdout_count": len(holdout),
        "holdout_dates": holdout_days,
        "history_date_count": len(history),
        "excluded_holdout_dates": holdout_days,
        "no_date_leakage": True,
        "holdout_source": "separate_dataset" if holdout_rows is not None else "sample_role",
    }
    report["data"] = report["dataset"]
    report["summary"] = (
        f"同批 {len(holdout)} 张留验试纸、{len(holdout_days)} 个试机日比较完成；"
        + ("达到换用门槛。" if report["eligible"] else "未达到换用门槛，应保留在用旧版。")
    )
    return report

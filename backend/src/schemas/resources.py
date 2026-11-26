from datetime import datetime, timezone


def utc(value: datetime | None):
    return value.replace(tzinfo=timezone.utc).isoformat() if value else None


def dataset_view(dataset, include_rows=False):
    item = {
        "id": dataset.id, "name": dataset.name, "count": len(dataset.rows),
        "checksum": dataset.checksum, "author": dataset.author, "created_at": utc(dataset.created_at),
    }
    if include_rows:
        item["rows"] = dataset.rows
    return item


def run_view(run, include_result=False):
    item = {
        "id": run.id, "dataset_id": run.dataset_id, "label": run.label, "parameter": run.parameter,
        "model_key": getattr(run, "model_key", "cross-ridge-v1"),
        "model_version": getattr(run, "model_key", "cross-ridge-v1"),
        "model_id": getattr(run, "model_key", "cross-ridge-v1"),
        "ordered_model": getattr(run, "model_key", "cross-ridge-v1"),
        "model_snapshot_at_submission": True,
        "model_fingerprint": getattr(run, "model_fingerprint", None),
        "status": run.status, "error": run.error, "actor": run.actor,
        "created_at": utc(run.created_at), "started_at": utc(run.started_at),
        "finished_at": utc(run.finished_at),
    }
    if include_result:
        item["result"] = run.result
    return item


def review_view(review):
    return {
        "id": review.id, "run_id": review.run_id, "decision": review.decision,
        "samples": review.samples, "note": review.note, "reviewer": review.reviewer,
        "created_at": utc(review.created_at),
    }


def audit_view(event):
    return {"actor": event.actor, "action": event.action, "subject": event.subject, "created_at": utc(event.created_at)}


def validation_view(item, include_report=False):
    view = {
        "id": item.id,
        "comparison_id": item.id,
        "dataset_id": item.dataset_id,
        "holdout_dataset_id": item.dataset_id,
        "training_dataset_id": getattr(item, "training_dataset_id", None),
        "label": item.label,
        "parameter": item.parameter,
        "base_model_key": item.base_model_key,
        "old_model_key": item.base_model_key,
        "base_fingerprint": item.base_fingerprint,
        "base_state_adoption_id": getattr(item, "base_state_adoption_id", None),
        "candidate_model_key": item.candidate_model_key,
        "new_model_key": item.candidate_model_key,
        "candidate_fingerprint": item.candidate_fingerprint,
        "eligible": item.eligible,
        "can_adopt": item.eligible,
        "actor": item.actor,
        "created_at": utc(item.created_at),
    }
    if include_report:
        view["report"] = item.report
    return view


def adoption_view(item):
    return {
        "id": item.id,
        "validation_id": item.validation_id,
        "comparison_id": item.validation_id,
        "idempotency_key": item.idempotency_key,
        "request_id": item.idempotency_key,
        "base_key": item.base_key,
        "old_model_key": item.base_key,
        "candidate_key": item.candidate_key,
        "new_model_key": item.candidate_key,
        "status": item.status,
        "actor": item.actor,
        "error": item.error,
        "created_at": utc(item.created_at),
        "applied_at": utc(item.applied_at),
    }

import logging

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.business.holdout_evaluation import evaluate_rows
from src.business.model_adoption import ModelAdoptionError
from src.business.model_recipes import DEFAULT_MODEL_KEY, MODEL_RECIPES, fingerprint, public_recipe
from src.config.settings import Session
from src.models.governance import ModelAdoption, ModelState, ModelValidation
from src.schemas.resources import adoption_view, validation_view
from src.services import datasets
from src.services.audit import record

logger = logging.getLogger(__name__)


def ensure_state(db, lock: bool = False):
    options = {"with_for_update": True} if lock else {}
    state = db.get(ModelState, "active", **options)
    if state is None:
        state = ModelState(
            id="active", active_key=DEFAULT_MODEL_KEY,
            active_fingerprint=fingerprint(DEFAULT_MODEL_KEY), updated_by="system",
        )
        db.add(state)
        db.flush()
    return state


def model_center():
    with Session() as db:
        state = ensure_state(db)
        active_key = state.active_key
        validations = db.scalars(
            select(ModelValidation).order_by(ModelValidation.created_at.desc(), ModelValidation.id.desc()).limit(30)
        ).all()
        adoptions = db.scalars(
            select(ModelAdoption).order_by(ModelAdoption.created_at.desc(), ModelAdoption.id.desc()).limit(30)
        ).all()
        return {
            "active_model": {
                **public_recipe(active_key),
                "updated_by": state.updated_by,
                "updated_at": state.updated_at.isoformat() if state.updated_at else None,
                "adoption_id": state.adoption_id,
            },
            "current_model_key": active_key,
            "models": [public_recipe(key) for key in MODEL_RECIPES],
            "validations": [validation_view(item) for item in validations],
            "comparisons": [validation_view(item) for item in validations],
            "adoptions": [adoption_view(item) for item in adoptions],
            "switches": [adoption_view(item) for item in adoptions],
        }


def list_validations():
    with Session() as db:
        items = db.scalars(
            select(ModelValidation).order_by(ModelValidation.created_at.desc(), ModelValidation.id.desc()).limit(100)
        )
        return [validation_view(item) for item in items]


def require_validation(db, validation_id):
    item = db.get(ModelValidation, validation_id)
    if item is None:
        raise HTTPException(404, "留验比较单不存在。")
    return item


def validation_details(validation_id):
    with Session() as db:
        return validation_view(require_validation(db, validation_id), include_report=True)


def _http_error(exc):
    status = getattr(exc, "status_code", 422)
    if isinstance(exc, ValueError) and not isinstance(exc, ModelAdoptionError):
        status = 422
    raise HTTPException(status, str(exc)) from exc


def create_validation(body, actor):
    explicit_training_id = (
        body.training_dataset_id or body.history_dataset_id
        or body.base_dataset_id or body.baseline_dataset_id
    )
    separate_holdout_id = body.holdout_dataset_id
    if separate_holdout_id:
        holdout_id = separate_holdout_id
        training_id = explicit_training_id or body.dataset_id
        if not training_id:
            raise HTTPException(422, "分开导入留验试纸时，还需要提供训练/基线数据版本。")
    else:
        holdout_id = None
        training_id = body.dataset_id or explicit_training_id
        if not training_id:
            raise HTTPException(422, "请选择包含留验试纸的数据版本。")
    parameter = body.parameter if body.parameter is not None else body.coverage_pct
    with Session.begin() as db:
        training_dataset = datasets.require(db, training_id)
        if holdout_id:
            holdout_dataset = datasets.require(db, holdout_id)
            primary_dataset = holdout_dataset
            holdout_rows = holdout_dataset.rows
        else:
            primary_dataset = training_dataset
            holdout_rows = None
        state = ensure_state(db)
        base_key = body.base_model_key or body.old_model_key or state.active_key
        candidate_key = body.candidate_model_key or body.new_model_key or "swift-cross-v2"
        try:
            report = evaluate_rows(
                training_dataset.rows,
                holdout_rows,
                base_model_key=base_key,
                candidate_model_key=candidate_key,
                parameter=parameter,
                base_parameter=body.base_parameter,
                candidate_parameter=body.candidate_parameter,
                holdout_dates=body.holdout_dates,
            )
            base_fp = fingerprint(base_key)
            candidate_fp = fingerprint(candidate_key)
        except (ValueError, ModelAdoptionError) as exc:
            _http_error(exc)
        item = ModelValidation(
            dataset_id=primary_dataset.id,
            training_dataset_id=training_dataset.id if holdout_id else None,
            label=body.label or "留验比较",
            parameter=parameter,
            base_model_key=base_key,
            base_fingerprint=base_fp,
            base_state_adoption_id=state.adoption_id,
            candidate_model_key=candidate_key,
            candidate_fingerprint=candidate_fp,
            eligible=bool(report["eligible"]),
            report=report,
            actor=actor,
        )
        db.add(item)
        db.flush()
        record(db, actor, "model_validation_created", item.id)
        response = validation_view(item, include_report=True)
    return response


def _find_adoption(db, *, key=None, validation_id=None):
    if key:
        item = db.scalar(select(ModelAdoption).where(ModelAdoption.idempotency_key == key))
        if item is not None:
            return item
    if validation_id:
        return db.scalar(select(ModelAdoption).where(ModelAdoption.validation_id == validation_id))
    return None


def adopt(body, actor, idempotency_key=None):
    key = (
        idempotency_key
        or getattr(body, "idempotency_key", None)
        or getattr(body, "request_id", None)
    )
    validation_id = getattr(body, "validation_id", None) or getattr(body, "comparison_id", None)
    if not validation_id:
        raise HTTPException(422, "缺少留验比较单编号。")

    try:
        with Session.begin() as db:
            existing = _find_adoption(db, key=key)
            if existing is not None:
                if validation_id and existing.validation_id != validation_id:
                    raise HTTPException(409, "幂等键已用于另一个换用请求，不能重复使用。")
                return adoption_view(existing)

            item = require_validation(db, validation_id)
            existing_for_item = _find_adoption(db, validation_id=validation_id)
            if existing_for_item is not None:
                return adoption_view(existing_for_item)

            state = ensure_state(db, lock=True)
            if not item.eligible:
                raise HTTPException(409, "留验证据未达到换用门槛，已保留在用旧版；请按工况补充留验或调整候选配方。")
            if state.active_key != item.base_model_key or state.active_fingerprint != item.base_fingerprint:
                raise HTTPException(409, "在用模型已变化，这份比较的旧模型基线已失效；请重新比较后再换用。")
            if (item.base_state_adoption_id or None) != (state.adoption_id or None):
                raise HTTPException(409, "在用模型刚发生过换用，原比较已失效；请按当前模型重新比较。")
            expected = getattr(body, "expected_active_model", None)
            if expected and expected != state.active_key:
                raise HTTPException(409, "下单后的在用模型与请求预期不一致，请刷新后重试。")

            from src.models.base import utcnow

            adoption = ModelAdoption(
                validation_id=item.id,
                idempotency_key=key or item.id,
                base_key=item.base_model_key,
                candidate_key=item.candidate_model_key,
                status="applied",
                actor=actor,
                applied_at=utcnow(),
            )
            db.add(adoption)
            db.flush()
            state.active_key = item.candidate_model_key
            state.active_fingerprint = item.candidate_fingerprint
            state.updated_by = actor
            state.adoption_id = adoption.id
            from src.models.base import utcnow as now2

            state.updated_at = now2()
            record(db, actor, "model_adopted", adoption.id)
            return adoption_view(adoption)
    except IntegrityError:
        logger.info("Adoption idempotency conflict for key=%s validation=%s", key, validation_id)
        with Session() as db:
            existing = _find_adoption(db, key=key)
            if existing is not None:
                if existing.validation_id != validation_id:
                    raise HTTPException(409, "幂等键已用于另一个换用请求，不能重复使用。")
                return adoption_view(existing)
            existing = _find_adoption(db, validation_id=validation_id)
            if existing is not None:
                return adoption_view(existing)
        raise HTTPException(409, "相同换用请求正在处理或已用于另一张比较单。")


def get_adoption(adoption_id):
    with Session() as db:
        item = db.get(ModelAdoption, adoption_id)
        if item is None:
            raise HTTPException(404, "换用记录不存在。")
        return adoption_view(item)


def adopt_path(validation_id, actor, idempotency_key=None):
    class PathAdoption:
        pass

    body = PathAdoption()
    body.validation_id = validation_id
    body.comparison_id = validation_id
    body.idempotency_key = idempotency_key
    body.request_id = None
    body.expected_active_model = None
    return adopt(body, actor, idempotency_key=idempotency_key or validation_id)

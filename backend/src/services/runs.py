import logging

from fastapi import HTTPException
from redis.exceptions import RedisError
from sqlalchemy import select

from src.business.model_recipes import require_recipe
from src.business.profile import PROFILE
from src.config.settings import Session, redis
from src.models.run import Run
from src.schemas.resources import run_view
from src.services import datasets, governance
from src.services.audit import record

WAKE_KEY = "workbench:wake"
logger = logging.getLogger(__name__)


def require(db, run_id):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "计算单不存在。")
    return run


def list_runs():
    with Session() as db:
        return [run_view(run) for run in db.scalars(select(Run).order_by(Run.created_at.desc(), Run.id.desc()))]


def submit(body, actor):
    control = PROFILE["parameter"]
    with Session.begin() as db:
        state = governance.ensure_state(db)
        active_recipe = require_recipe(state.active_key)
        parameter = body.parameter if body.parameter is not None else active_recipe["default_parameter"]
        if not control["min"] <= parameter <= control["max"]:
            raise HTTPException(422, f"「{control['label']}」超出允许范围。")
        datasets.require(db, body.dataset_id)
        # Snapshot both the parameter and the active model when the order is
        # placed.  A later adoption must not rewrite queued calculations.
        run = Run(
            dataset_id=body.dataset_id, label=body.label, parameter=parameter, actor=actor,
            model_key=state.active_key, model_fingerprint=state.active_fingerprint,
        )
        db.add(run)
        db.flush()
        record(db, actor, "run_queued", run.id)
        response = run_view(run)
    try:
        with redis.pipeline() as commands:
            commands.lpush(WAKE_KEY, run.id)
            commands.ltrim(WAKE_KEY, 0, 99)
            commands.execute()
    except RedisError:
        logger.warning("Redis wake signal failed for %s. Worker will poll the database.", run.id)
    return response

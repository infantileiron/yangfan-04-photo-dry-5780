"""Initial schema for a fresh Compose volume and additive columns."""
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from src.models.account import Account
from src.models.audit import Event
from src.models.dataset import Dataset
from src.models.governance import ModelAdoption, ModelState, ModelValidation
from src.models.review import Review
from src.models.run import Run

SCHEMA = (Account, Dataset, Run, Review, Event, ModelState, ModelValidation, ModelAdoption)


def migrate(engine):
    tables = [model.__table__ for model in SCHEMA]
    Account.metadata.create_all(engine, tables=tables)
    inspector = inspect(engine)
    run_columns = {column["name"] for column in inspector.get_columns("runs")}
    statements = []
    if "model_key" not in run_columns:
        statements.append("ALTER TABLE runs ADD COLUMN model_key VARCHAR(64) NOT NULL DEFAULT 'cross-ridge-v1'")
    if "model_fingerprint" not in run_columns:
        statements.append("ALTER TABLE runs ADD COLUMN model_fingerprint VARCHAR(64) NULL")
    validation_columns = {column["name"] for column in inspector.get_columns("model_validations")} if "model_validations" in inspector.get_table_names() else set()
    if "model_validations" in inspector.get_table_names() and "base_state_adoption_id" not in validation_columns:
        statements.append("ALTER TABLE model_validations ADD COLUMN base_state_adoption_id VARCHAR(36) NULL")
    if "model_validations" in inspector.get_table_names() and "training_dataset_id" not in validation_columns:
        statements.append("ALTER TABLE model_validations ADD COLUMN training_dataset_id VARCHAR(36) NULL")
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
    # SQLite and MySQL both accept the additive ALTER statements above.
    with Session(engine) as db:
        if db.get(ModelState, "active") is None:
            from src.business.model_recipes import DEFAULT_MODEL_KEY, fingerprint

            db.add(ModelState(
                id="active", active_key=DEFAULT_MODEL_KEY,
                active_fingerprint=fingerprint(DEFAULT_MODEL_KEY), updated_by="system",
            ))
            db.commit()

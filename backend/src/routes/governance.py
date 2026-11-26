from fastapi import APIRouter, Depends, Header

from src.routes.dependencies import current_account, model_operator
from src.schemas.requests import AdoptionRequest, ValidationRequest
from src.services import governance

router = APIRouter(dependencies=[Depends(current_account)])


@router.get("/model-center")
@router.get("/model-state")
@router.get("/active-model")
@router.get("/models/active")
def model_center():
    return governance.model_center()


@router.get("/models")
def models():
    return governance.model_center()


for path in (
    "/validations", "/model-validations", "/holdout-evaluations", "/holdout-comparisons",
    "/holdouts", "/model-evaluations", "/comparisons",
):
    router.add_api_route(path, governance.list_validations, methods=["GET"])


@router.post("/validations", status_code=201)
@router.post("/model-validations", status_code=201)
@router.post("/holdout-evaluations", status_code=201)
@router.post("/holdout-comparisons", status_code=201)
@router.post("/holdouts", status_code=201)
@router.post("/model-evaluations", status_code=201)
@router.post("/comparisons", status_code=201)
def create_validation(body: ValidationRequest, account=Depends(model_operator)):
    return governance.create_validation(body, account.email)


@router.get("/validations/{validation_id}")
@router.get("/model-validations/{validation_id}")
@router.get("/holdout-evaluations/{validation_id}")
@router.get("/holdout-comparisons/{validation_id}")
@router.get("/holdouts/{validation_id}")
@router.get("/model-evaluations/{validation_id}")
@router.get("/comparisons/{validation_id}")
def validation_details(validation_id: str):
    return governance.validation_details(validation_id)


@router.post("/model-adoptions", status_code=201)
@router.post("/model-switches", status_code=201)
@router.post("/models/switch", status_code=201)
@router.post("/adoptions", status_code=201)
@router.post("/models/adopt", status_code=201)
def adopt_model(
    body: AdoptionRequest,
    account=Depends(model_operator),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    x_idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
):
    return governance.adopt(body, account.email, idempotency_key or x_idempotency_key)


@router.post("/validations/{validation_id}/adopt", status_code=201)
@router.post("/model-validations/{validation_id}/adopt", status_code=201)
@router.post("/comparisons/{validation_id}/adopt", status_code=201)
@router.post("/model-evaluations/{validation_id}/adopt", status_code=201)
def adopt_validation_path(
    validation_id: str,
    account=Depends(model_operator),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    return governance.adopt_path(validation_id, account.email, idempotency_key)


@router.get("/model-adoptions/{adoption_id}")
def adoption_details(adoption_id: str):
    return governance.get_adoption(adoption_id)


@router.post("/model-adoptions/{adoption_id}/retry", status_code=200)
def retry_adoption(adoption_id: str, account=Depends(model_operator)):
    return governance.get_adoption(adoption_id)

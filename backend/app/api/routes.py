from fastapi import APIRouter

from app.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    return PredictionResponse(
        prediction=0,
        risk_label="low",
        probability=0.50,
        message="Placeholder prediction. ML model not integrated yet.",
    )
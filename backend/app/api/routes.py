from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import PredictionRecord
from app.db.session import get_db
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import predict_diabetes_risk

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/predictions")
def list_predictions(db: Session = Depends(get_db)):
    records = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).all()

    return [
        {
            "id": record.id,
            "pregnancies": record.pregnancies,
            "glucose": record.glucose,
            "blood_pressure": record.blood_pressure,
            "skin_thickness": record.skin_thickness,
            "insulin": record.insulin,
            "bmi": record.bmi,
            "diabetes_pedigree_function": record.diabetes_pedigree_function,
            "age": record.age,
            "prediction": record.prediction,
            "risk_label": record.risk_label,
            "probability": record.probability,
            "message": record.message,
            "model_version": record.model_version,
            "created_at": record.created_at,
        }
        for record in records
    ]


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, db: Session = Depends(get_db)):
    result = predict_diabetes_risk(payload)

    response = PredictionResponse(
        prediction=result["prediction"],
        risk_label=result["risk_label"],
        probability=result["probability"],
        message=result["message"],
    )

    record = PredictionRecord(
        pregnancies=payload.pregnancies,
        glucose=payload.glucose,
        blood_pressure=payload.blood_pressure,
        skin_thickness=payload.skin_thickness,
        insulin=payload.insulin,
        bmi=payload.bmi,
        diabetes_pedigree_function=payload.diabetes_pedigree_function,
        age=payload.age,
        prediction=response.prediction,
        risk_label=response.risk_label,
        probability=response.probability,
        message=response.message,
        model_version="logistic-regression-v1",
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return response
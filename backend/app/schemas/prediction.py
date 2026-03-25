from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: int = Field(..., ge=0, le=300)
    blood_pressure: int = Field(..., ge=0, le=200)
    skin_thickness: int = Field(..., ge=0, le=100)
    insulin: int = Field(..., ge=0, le=1000)
    bmi: float = Field(..., ge=0, le=100)
    diabetes_pedigree_function: float = Field(..., ge=0, le=5)
    age: int = Field(..., ge=1, le=120)


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    probability: float
    message: str
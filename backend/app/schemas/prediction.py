from pydantic import BaseModel

class PredictionRequest(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree_function: float
    age: int


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    probability: float
    message: str
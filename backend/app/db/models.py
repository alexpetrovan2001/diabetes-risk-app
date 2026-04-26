from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.db.database import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    pregnancies = Column(Integer, nullable=False)
    glucose = Column(Integer, nullable=False)
    blood_pressure = Column(Integer, nullable=False)
    skin_thickness = Column(Integer, nullable=False)
    insulin = Column(Integer, nullable=False)
    bmi = Column(Float, nullable=False)
    diabetes_pedigree_function = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    prediction = Column(Integer, nullable=False)
    risk_label = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    message = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
# Data Contract

## Prediction Request
The frontend will send these fields to the backend:

- pregnancies
- glucose
- blood_pressure
- skin_thickness
- insulin
- bmi
- diabetes_pedigree_function
- age

Example request:

{
  "pregnancies": 0,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 79,
  "bmi": 25.6,
  "diabetes_pedigree_function": 0.45,
  "age": 33
}

## Prediction Response
The backend will return:

{
  "prediction": 1,
  "risk_label": "high",
  "probability": 0.78,
  "message": "Predicted elevated diabetes risk"
}

## Database Record
Each stored prediction record should include:
- id
- pregnancies
- glucose
- blood_pressure
- skin_thickness
- insulin
- bmi
- diabetes_pedigree_function
- age
- prediction
- risk_label
- probability
- created_at
- model_version
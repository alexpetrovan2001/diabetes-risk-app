# Project Scope

## Goal
Build a local proof-of-concept web application that predicts diabetes risk
using a machine learning model and stores prediction results in a database.

## Type of Application
Educational and research POC only.
Not a medical diagnosis system.

## Main User Flow
1. User opens the application
2. User enters medical input values
3. Application sends data to backend
4. Backend runs prediction using trained ML model
5. Application shows prediction result
6. Backend stores input data and prediction result in database
7. User can review previous prediction records

## Input Data
Planned input fields:
- pregnancies
- glucose
- blood_pressure
- skin_thickness
- insulin
- bmi
- diabetes_pedigree_function
- age

## Output
Planned prediction output:
- diabetes risk class
- optional probability score

## Data to Store
For each prediction, store:
- input values
- prediction result
- probability score if available
- timestamp
- optional model version

## Initial Database Choice
SQLite for local development and POC stage.

## Initial ML Scope
Use a supervised classification model trained on a public dataset.

## Out of Scope
- real clinical diagnosis
- authentication and user accounts
- advanced reporting dashboards
- cloud deployment in the first stage
- production-grade security and compliance

## Success Criteria for First Working Version
- user can submit input data
- backend returns a prediction
- prediction is stored in database
- stored predictions can be viewed
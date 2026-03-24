# Project Notes

## Project Title
Sistem inteligent pentru predicția riscului de diabet folosind tehnici de
învățare automată și stocarea rezultatelor într-o bază de date

## Project Type
Educational and research proof of concept (POC)

## Purpose
The purpose of this project is to build a prototype application that can
estimate diabetes risk based on relevant medical input data and store the
prediction results in a database for later review and analysis.

## Main Objectives
- collect medical input data through a simple user interface
- run a machine learning model that predicts diabetes risk
- display the prediction result to the user
- store the input data and prediction result in a database
- support later consultation and analysis of saved results

## Installed Tools Version
- Git - 2.50.3.windows.2
- Python - 3.14.3
- Node.js - v24.14.1 - and npm - 11.12.0

## Initial Project Structure
- `backend/` - backend application and machine learning logic
- `frontend/` - user interface
- `data/` - datasets used for training/testing
- `models/` - exported trained models
- `docs/` - project notes and planning documents
- `scripts/` - utility scripts

## Initial Dependencies
### Backend
- FastAPI
- Uvicorn
- pandas
- scikit-learn
- numpy
- joblib

### Frontend
- React
- Vite

## Scope of the POC
The prototype should:
- allow input of relevant medical parameters
- generate a diabetes risk prediction
- store prediction results in a database
- allow review of stored prediction records

The prototype will not:
- provide medical diagnosis
- replace medical consultation
- be used in clinical production
- process sensitive real patient data unless explicitly allowed and handled
  correctly

## Research and Ethics Notes
- this application is for educational and research purposes
- predictions are informational only
- results must not be interpreted as medical diagnosis
- the project should preferably use public, anonymized, or synthetic datasets
- personal or identifiable patient data should be avoided in the POC

## Development Decisions So Far
- the project will be developed locally on a personal computer
- deployment can be done later after the local version works
- the system will include a frontend, backend, ML model, and database
- the first priority is a functional end-to-end prototype

## Next Steps
- define exact application scope and features
- choose the data fields used for prediction
- decide what will be stored in the database
- set up the backend structure
- set up the frontend structure
- integrate the machine learning workflow
- prepare local database storage
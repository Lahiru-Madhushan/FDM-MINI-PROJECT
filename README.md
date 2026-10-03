# FDM-MINI-PROJECT

Telco customer churn prediction (IT3051 Fundamentals of Data Mining).

## Churn Risk Checker – running the app

The backend (`backend/`) is a FastAPI service that loads the saved preprocessing pipeline and the final
XGBoost model from `models/`. It also serves the frontend (`frontend/`), so one command starts the whole system.

```powershell
# from the project root, once
.venv\Scripts\python -m pip install -r backend\requirements.txt

# start the app
.venv\Scripts\python -m uvicorn backend.app.main:app --reload
```

Open **http://127.0.0.1:8000** for the app, or **http://127.0.0.1:8000/docs** for the interactive API documentation.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service status, model version, decision threshold and test metrics |
| `GET /api/schema` | Input fields, allowed values and ranges (used to build the form) |
| `POST /api/predict` | Predict one customer |
| `POST /api/predict/batch` | Predict up to 1,000 customers (JSON) |
| `POST /api/predict/csv` | Score a CSV file (dataset or API column names; up to 5,000 rows) |

Each prediction returns the churn probability, the decision (threshold 0.51), a risk level, the factors that raise
and lower this customer's risk, suggested retention actions, and warnings (e.g. when an optional value was imputed).
Invalid or inconsistent inputs return HTTP 422 with one message per field.

Run the tests (including a check that the API reproduces the notebooks' predictions for all 1,409 test customers):

```powershell
.venv\Scripts\python -m pytest backend\tests -q
```

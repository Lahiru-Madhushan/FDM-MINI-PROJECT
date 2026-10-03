"""FastAPI service for the Telco churn prediction model.

Run from the project root:
    .venv\\Scripts\\python -m uvicorn backend.app.main:app --reload
then open http://127.0.0.1:8000
"""
import io
import json
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import FRONTEND_DIR, MAX_CSV_BYTES, MAX_CSV_ROWS, MODEL_DIR
from .predictor import ChurnPredictor
from .schemas import (COLUMN_MAP, BatchRequest, BatchResponse, CustomerInput, PredictionResult, check_consistency,
                      field_schema)

app = FastAPI(title="Telco Customer Churn Prediction API", version="1.0.0",
              description="Predicts whether a telecom customer is likely to churn, using the project's final "
                          "tuned XGBoost model and the same preprocessing used during model development.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@lru_cache
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor(MODEL_DIR)


# ---------------------------------------------------------------------- error handling
def _field_name(loc: tuple) -> str:
    parts = [str(p) for p in loc if p != "body"]
    name = ""
    for p in parts:
        name += f"[{p}]" if p.isdigit() else (f".{p}" if name else p)
    return name or "body"


def _pydantic_errors(errors: list[dict]) -> list[dict]:
    return [{"field": _field_name(e["loc"]), "message": e["msg"]} for e in errors]


def invalid_input(errors: list[dict]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return invalid_input(_pydantic_errors(exc.errors()))


# ---------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "model": get_predictor().info()}


@app.get("/api/schema")
def schema():
    """Input fields, allowed values and ranges – used by the frontend to build and validate its form."""
    return {"fields": field_schema(), "decision_threshold": get_predictor().threshold}


@app.post("/api/predict", response_model=PredictionResult)
def predict(customer: CustomerInput):
    """Predict churn for one customer."""
    errors = check_consistency(customer)
    if errors:
        return invalid_input(errors)
    return get_predictor().predict([customer.model_dump()])[0]


def _score_many(customers: list[CustomerInput], ids: list, row_errors: list[dict]) -> dict:
    valid, valid_idx = [], []
    for i, c in enumerate(customers):
        if c is None:
            continue
        errs = check_consistency(c)
        if errs:
            row_errors.append({"row": i + 1, "customer_id": ids[i], "errors": errs})
        else:
            valid.append(c.model_dump())
            valid_idx.append(i)
    results = get_predictor().predict(valid) if valid else []
    items = [{**r, "index": i + 1, "customer_id": ids[i]} for r, i in zip(results, valid_idx)]
    return {"count": len(items), "at_risk_count": sum(r["at_risk"] for r in items), "results": items,
            "errors": sorted(row_errors, key=lambda e: e["row"])}


@app.post("/api/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Predict churn for up to 1,000 customers. Rows that break a consistency rule are reported in `errors`."""
    return _score_many(request.customers, [None] * len(request.customers), [])


@app.post("/api/predict/csv", response_model=BatchResponse)
async def predict_csv(file: UploadFile = File(...)):
    """Score a CSV file. Columns may use the API names (e.g. `monthly_charge`) or the original dataset names
    (e.g. `Monthly Charge`); extra columns are ignored and a `Customer ID` column is kept as an identifier."""
    content = await file.read()
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_CSV_BYTES // (1024 * 1024)} MB")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        return invalid_input([{"field": "file", "message": "Could not read the file as CSV."}])
    if df.empty:
        return invalid_input([{"field": "file", "message": "The CSV file has no rows."}])
    if len(df) > MAX_CSV_ROWS:
        return invalid_input([{"field": "file", "message": f"At most {MAX_CSV_ROWS} rows can be scored at once."}])

    dataset_to_api = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns={c: dataset_to_api.get(c.strip(), c.strip()) for c in df.columns})
    id_col = next((c for c in ["Customer ID", "customer_id"] if c in df.columns), None)
    missing = [c for c in ["gender", "age", "contract", "tenure_in_months", "monthly_charge"] if c not in df.columns]
    if missing:
        return invalid_input([{"field": "file", "message": "Missing required columns: " + ", ".join(missing)}])
    # Same cleaning rule as Data_Cleaning.ipynb: a blank Offer / Internet Type means none
    if "offer" in df.columns:
        df["offer"] = df["offer"].fillna("No Offer")
    if "internet_type" in df.columns:
        df["internet_type"] = df["internet_type"].fillna("No Internet Service")

    records = json.loads(df[[c for c in df.columns if c in COLUMN_MAP]].to_json(orient="records"))
    ids = [str(v) for v in df[id_col]] if id_col else [None] * len(df)
    customers, row_errors = [], []
    for i, rec in enumerate(records):
        try:
            customers.append(CustomerInput(**rec))
        except ValidationError as e:
            customers.append(None)
            row_errors.append({"row": i + 1, "customer_id": ids[i], "errors": _pydantic_errors(e.errors())})
    return _score_many(customers, ids, row_errors)


# The frontend is served from the same server, so the whole system runs with one command
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

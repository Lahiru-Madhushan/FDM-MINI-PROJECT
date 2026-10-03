"""API tests. Run from the project root:  .venv\\Scripts\\python -m pytest backend/tests -q"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from backend.app.main import app
from backend.app.schemas import COLUMN_MAP, CustomerInput, FIELD_SPECS

ROOT = Path(__file__).resolve().parents[2]
client = TestClient(app)
API_NAME = {v: k for k, v in COLUMN_MAP.items()}


def payload_from_raw(row: pd.Series) -> dict:
    d = {API_NAME[col]: row[col] for col in API_NAME}
    if not isinstance(d["offer"], str):
        d["offer"] = "No Offer"
    if not isinstance(d["internet_type"], str):
        d["internet_type"] = "No Internet Service"
    return json.loads(pd.Series(d).to_json())


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(ROOT / "Dataset/telco.csv")


@pytest.fixture(scope="module")
def test_split(raw):
    fs = pd.read_csv(ROOT / "Notebook/Preprocessing/feature_selected_data.csv")
    _, test_idx = train_test_split(fs.index, test_size=0.2, random_state=42, stratify=fs["Churn Label"])
    return raw.loc[test_idx]


@pytest.fixture
def customer(raw):
    return payload_from_raw(raw.iloc[0])          # month-to-month, tenure 1 → high risk


# ---------------------------------------------------------------- service
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model"]["decision_threshold"] == 0.51


def test_schema_matches_input_model():
    names = [f["name"] for f in client.get("/api/schema").json()["fields"]]
    assert names == [s[1] for s in FIELD_SPECS]
    assert set(names) == set(CustomerInput.model_fields)


def test_frontend_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "Churn Risk Checker" in r.text


# ---------------------------------------------------------------- valid predictions
def test_predict_returns_clear_result(customer):
    r = client.post("/api/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["at_risk"] == (body["churn_probability"] >= body["threshold"])
    assert body["prediction"] in ("Likely to churn", "Likely to stay")
    assert body["factors_increasing_risk"] and body["recommendations"]


def test_api_reproduces_notebook_predictions(test_split):
    """End-to-end: raw test customers through the API give exactly the notebook pipeline's predictions."""
    payloads = [payload_from_raw(row) for _, row in test_split.iterrows()]
    api_proba = []
    for start in range(0, len(payloads), 1000):          # the batch endpoint accepts up to 1,000 customers
        r = client.post("/api/predict/batch", json={"customers": payloads[start:start + 1000]})
        assert r.status_code == 200 and not r.json()["errors"]
        api_proba += [x["churn_probability"] for x in r.json()["results"]]
    api_proba = np.array(api_proba)
    assert len(api_proba) == 1409

    test_df = pd.read_csv(ROOT / "test_processed.csv")
    model = joblib.load(ROOT / "models/xgboost_final.pkl")
    nb_proba = model.predict_proba(test_df.drop(columns="Churn Label"))[:, 1]
    assert np.abs(api_proba - nb_proba).max() < 1e-4                 # API rounds to 4 decimals
    assert round(f1_score(test_df["Churn Label"], nb_proba >= 0.51), 4) == 0.7206


def test_missing_optional_numbers_are_imputed(customer):
    for f in ["cltv", "total_refunds"]:
        customer.pop(f)
    body = client.post("/api/predict", json=customer).json()
    assert any("CLTV was not provided" in w for w in body["warnings"])


def test_out_of_training_range_warns(customer):
    customer["age"] = 95
    body = client.post("/api/predict", json=customer).json()
    assert any("outside the range seen in training" in w for w in body["warnings"])


def test_no_phone_service_fills_dependent_fields(customer):
    customer.update(phone_service="No", multiple_lines=None, avg_monthly_long_distance_charges=None)
    assert client.post("/api/predict", json=customer).status_code == 200


# ---------------------------------------------------------------- invalid inputs
@pytest.mark.parametrize("field, value", [
    ("age", 10), ("age", "old"), ("tenure_in_months", 0), ("tenure_in_months", 100), ("monthly_charge", -5),
    ("contract", "Weekly"), ("gender", "X"), ("number_of_referrals", 2.5),
])
def test_invalid_values_rejected(customer, field, value):
    customer[field] = value
    r = client.post("/api/predict", json=customer)
    assert r.status_code == 422
    assert r.json()["errors"][0]["field"] == field


def test_missing_required_field(customer):
    del customer["contract"]
    r = client.post("/api/predict", json=customer)
    assert r.status_code == 422 and r.json()["errors"] == [{"field": "contract", "message": "Field required"}]


def test_unknown_field_rejected(customer):
    customer["satisfaction_score"] = 5                # leakage column must not be accepted
    assert client.post("/api/predict", json=customer).status_code == 422


def test_inconsistent_inputs_rejected(customer):
    customer.update(internet_type="No Internet Service", streaming_tv="Yes")
    r = client.post("/api/predict", json=customer)
    assert r.status_code == 422
    assert "streaming_tv" in [e["field"] for e in r.json()["errors"]]


def test_internet_addons_required_with_internet(customer):
    customer["internet_type"] = "DSL"
    customer.pop("online_security")
    r = client.post("/api/predict", json=customer)
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "online_security"


def test_batch_size_limit(customer):
    r = client.post("/api/predict/batch", json={"customers": [customer] * 1001})
    assert r.status_code == 422


def test_malformed_json():
    r = client.post("/api/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ---------------------------------------------------------------- batch / CSV
def test_csv_with_dataset_columns(test_split):
    cols = ["Customer ID"] + list(COLUMN_MAP.values())
    csv = test_split[cols].head(20).to_csv(index=False)       # blank Offer / Internet Type as in the raw data
    r = client.post("/api/predict/csv", files={"file": ("customers.csv", csv, "text/csv")})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 20 and not body["errors"]
    assert body["results"][0]["customer_id"] == test_split["Customer ID"].iloc[0]


def test_csv_reports_bad_rows(test_split):
    df = test_split[list(COLUMN_MAP.values())].head(5).copy()
    df.loc[df.index[1], "Age"] = 300
    df.loc[df.index[3], "Contract"] = "Forever"
    r = client.post("/api/predict/csv", files={"file": ("c.csv", df.to_csv(index=False), "text/csv")})
    body = r.json()
    assert body["count"] == 3 and [e["row"] for e in body["errors"]] == [2, 4]


def test_csv_missing_columns():
    r = client.post("/api/predict/csv", files={"file": ("c.csv", "a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422 and "Missing required columns" in r.json()["errors"][0]["message"]


def test_csv_not_a_csv():
    r = client.post("/api/predict/csv", files={"file": ("c.csv", io.BytesIO(b"\x00\xff\x00"), "text/csv")})
    assert r.status_code == 422

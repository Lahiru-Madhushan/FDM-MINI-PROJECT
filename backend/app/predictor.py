"""Loads the saved preprocessing pipeline + final XGBoost model and turns customer records into predictions."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .schemas import COLUMN_MAP, OPTIONAL_NUMERIC

# The 11 services counted by the engineered "Total Services" feature (Feature_Selection.ipynb)
SERVICE_COLUMNS = ["Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
                   "Device Protection Plan", "Premium Tech Support", "Streaming TV", "Streaming Movies",
                   "Streaming Music", "Unlimited Data"]

# Values outside these ranges were never seen in training (Dataset/telco.csv), so predictions are less reliable
TRAINING_RANGES = {"age": (19, 80), "number_of_dependents": (0, 9), "number_of_referrals": (0, 11),
                   "monthly_charge": (18.25, 118.75), "cltv": (2003, 6500), "avg_monthly_gb_download": (0, 85),
                   "avg_monthly_long_distance_charges": (0, 49.99), "total_refunds": (0, 49.79),
                   "total_extra_data_charges": (0, 150)}

LOW_RISK_BELOW = 0.30

# Retention actions for risk factors the retention team can act on
ACTIONS = {
    "Contract": "Offer an incentive (e.g. a discount) to move from a month-to-month to a one- or two-year contract.",
    "Tenure in Months": "New customer: schedule an early check-in call and onboarding support.",
    "Tenure Group": "New customer: schedule an early check-in call and onboarding support.",
    "Number of Referrals": "Invite the customer to the referral programme – customers who refer others rarely leave.",
    "Monthly Charge": "Review the plan price: consider a loyalty discount or a better-value bundle.",
    "Internet Type": "Check the customer's internet service quality and match competitor offers.",
    "Offer": "Review the customer's promotional offer – customers on Offer A or B churn the least.",
    "Payment Method": "Encourage switching to automatic credit-card payment.",
    "Online Security": "Offer a free trial of Online Security.",
    "Premium Tech Support": "Offer a free trial of Premium Tech Support.",
    "Paperless Billing": "Check the customer is happy with their billing experience.",
    "Total Services": "Suggest a service bundle that fits the customer's needs.",
}


class ChurnPredictor:
    def __init__(self, model_dir: Path):
        self.pipeline = joblib.load(model_dir / "preprocessing_pipeline.pkl")
        self.model = joblib.load(model_dir / "xgboost_final.pkl")
        with open(model_dir / "xgboost_final_metadata.json") as f:
            self.metadata = json.load(f)
        self.threshold = float(self.metadata["decision_threshold"])
        self.input_columns = list(self.pipeline.feature_names_in_)          # 29 selected features
        self.encoded_columns = list(self.pipeline.get_feature_names_out())  # 57 model columns
        if self.encoded_columns != self.metadata["features"]:
            raise RuntimeError("Preprocessing pipeline and model expect different feature columns")
        self.booster = self.model.get_booster()
        self.medians = self._training_medians()
        self.encoded_to_feature = self._map_encoded_columns()

    # ------------------------------------------------------------------ setup helpers
    def _training_medians(self) -> dict:
        num_name, num_pipe, num_cols = self.pipeline.transformers_[0]
        imputer = num_pipe.named_steps["imputer"]
        return dict(zip(num_cols, imputer.statistics_))

    def _map_encoded_columns(self) -> list[str]:
        """Original feature behind each of the 57 encoded columns (e.g. cat__Contract_One Year → Contract)."""
        _, cat_pipe, cat_cols = self.pipeline.transformers_[1]
        encoder = cat_pipe.named_steps["encoder"]
        mapping = {}
        for col, cats in zip(cat_cols, encoder.categories_):
            for cat in cats:
                mapping[f"cat__{col}_{cat}"] = col
        return [mapping.get(c, c.replace("num__", "")) for c in self.encoded_columns]

    # ------------------------------------------------------------------ preprocessing
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        """Row-level cleaning + feature engineering, identical to Data_Cleaning and Feature_Selection notebooks."""
        df = df.copy()
        df["Offer"] = df["Offer"].fillna("No Offer")
        df["Internet Type"] = df["Internet Type"].fillna("No Internet Service")
        df["Internet Service"] = np.where(df["Internet Type"] == "No Internet Service", "No", "Yes")
        df["Total Services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
        df["Tenure Group"] = pd.cut(df["Tenure in Months"], bins=[0, 12, 24, 48, 72],
                                    labels=["0-12", "13-24", "25-48", "49-72"], include_lowest=True).astype(str)
        return df

    def to_frame(self, customers: list[dict]) -> pd.DataFrame:
        """API records (snake_case) → DataFrame with the dataset's column names."""
        df = pd.DataFrame([{COLUMN_MAP[k]: v for k, v in c.items()} for c in customers])
        for field in OPTIONAL_NUMERIC:  # missing values (None) → NaN, which the pipeline's imputer fills
            df[COLUMN_MAP[field]] = pd.to_numeric(df[COLUMN_MAP[field]])
        return df

    # ------------------------------------------------------------------ prediction
    def predict(self, customers: list[dict], top_n: int = 3) -> list[dict]:
        features = self.prepare_features(self.to_frame(customers))[self.input_columns]
        encoded = pd.DataFrame(self.pipeline.transform(features), columns=self.encoded_columns)
        proba = self.model.predict_proba(encoded)[:, 1]
        # Per-customer SHAP contributions (log-odds) from XGBoost; last column is the bias term
        contribs = self.booster.predict(xgb.DMatrix(encoded, feature_names=self.encoded_columns),
                                        pred_contribs=True)[:, :-1]
        per_feature = pd.DataFrame(contribs, columns=self.encoded_to_feature).T.groupby(level=0).sum().T

        results = []
        for i, customer in enumerate(customers):
            p = float(proba[i])
            impacts = per_feature.iloc[i].sort_values()
            row = features.iloc[i]
            increasing = [self._factor(f, row[f], v) for f, v in impacts[::-1].items() if v > 0][:top_n]
            decreasing = [self._factor(f, row[f], v) for f, v in impacts.items() if v < 0][:top_n]
            at_risk = p >= self.threshold
            results.append({
                "churn_probability": round(p, 4),
                "churn_probability_pct": f"{p * 100:.1f}%",
                "at_risk": at_risk,
                "prediction": "Likely to churn" if at_risk else "Likely to stay",
                "risk_level": "High" if at_risk else ("Moderate" if p >= LOW_RISK_BELOW else "Low"),
                "threshold": self.threshold,
                "factors_increasing_risk": increasing,
                "factors_decreasing_risk": decreasing,
                "recommendations": self._recommendations(at_risk, increasing),
                "warnings": self._warnings(customer),
            })
        return results

    @staticmethod
    def _factor(feature: str, value, impact: float) -> dict:
        if isinstance(value, (float, np.floating)):
            value = f"{value:g}"
        return {"feature": feature, "value": str(value), "impact": round(float(impact), 4)}

    @staticmethod
    def _recommendations(at_risk: bool, increasing: list[dict]) -> list[str]:
        actions = []
        for f in increasing:
            action = ACTIONS.get(f["feature"])
            if action and action not in actions:
                actions.append(action)
        if not at_risk:
            return ["No urgent action needed – continue standard engagement."] + actions[:1]
        return actions or ["Contact the customer to understand their needs and offer a tailored retention deal."]

    def _warnings(self, customer: dict) -> list[str]:
        warnings = []
        for field in OPTIONAL_NUMERIC:
            if customer.get(field) is None:
                median = self.medians[COLUMN_MAP[field]]
                warnings.append(f"{COLUMN_MAP[field]} was not provided – the training median ({median:g}) was used.")
        for field, (lo, hi) in TRAINING_RANGES.items():
            v = customer.get(field)
            if v is not None and not lo <= v <= hi:
                warnings.append(f"{COLUMN_MAP[field]} = {v:g} is outside the range seen in training "
                                f"({lo:g}–{hi:g}); the prediction may be less reliable.")
        return warnings

    def info(self) -> dict:
        return {"model": self.metadata["model"], "xgboost_version": self.metadata["xgboost_version"],
                "decision_threshold": self.threshold, "test_metrics": self.metadata["test_metrics"],
                "input_features": len(self.input_columns), "model_columns": len(self.encoded_columns)}

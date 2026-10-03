"""Request / response models and input validation for the churn prediction API.

Field names are snake_case in the API; COLUMN_MAP translates them to the dataset column
names the preprocessing pipeline was trained on.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
OFFERS = ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
INTERNET_TYPES = ["Fiber Optic", "Cable", "DSL", "No Internet Service"]
CONTRACTS = ["Month-to-Month", "One Year", "Two Year"]
PAYMENT_METHODS = ["Bank Withdrawal", "Credit Card", "Mailed Check"]
NO_INTERNET = "No Internet Service"

PHONE_DEPENDENT = ["multiple_lines", "avg_monthly_long_distance_charges"]
INTERNET_ADDONS = ["online_security", "online_backup", "device_protection_plan", "premium_tech_support",
                   "streaming_tv", "streaming_movies", "streaming_music", "unlimited_data"]
INTERNET_NUMERIC = ["avg_monthly_gb_download", "total_extra_data_charges"]
# Numeric fields that may be left blank: the pipeline fills them with the training median
OPTIONAL_NUMERIC = ["avg_monthly_long_distance_charges", "avg_monthly_gb_download", "total_refunds",
                    "total_extra_data_charges", "cltv"]


class CustomerInput(BaseModel):
    """One customer, described with the attributes the company knows before churn."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Customer profile
    gender: Literal["Male", "Female"]
    age: int = Field(ge=18, le=100)
    married: YesNo
    number_of_dependents: int = Field(ge=0, le=20)
    # Account
    tenure_in_months: int = Field(ge=1, le=72)
    contract: Literal["Month-to-Month", "One Year", "Two Year"]
    offer: Literal["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"] = "No Offer"
    number_of_referrals: int = Field(ge=0, le=50)
    cltv: Optional[float] = Field(default=None, ge=0, le=20000)
    # Phone service
    phone_service: YesNo
    multiple_lines: Optional[YesNo] = None
    avg_monthly_long_distance_charges: Optional[float] = Field(default=None, ge=0, le=200)
    # Internet service
    internet_type: Literal["Fiber Optic", "Cable", "DSL", "No Internet Service"]
    avg_monthly_gb_download: Optional[float] = Field(default=None, ge=0, le=500)
    online_security: Optional[YesNo] = None
    online_backup: Optional[YesNo] = None
    device_protection_plan: Optional[YesNo] = None
    premium_tech_support: Optional[YesNo] = None
    streaming_tv: Optional[YesNo] = None
    streaming_movies: Optional[YesNo] = None
    streaming_music: Optional[YesNo] = None
    unlimited_data: Optional[YesNo] = None
    # Billing
    monthly_charge: float = Field(ge=0, le=500)
    paperless_billing: YesNo
    payment_method: Literal["Bank Withdrawal", "Credit Card", "Mailed Check"]
    total_refunds: Optional[float] = Field(default=None, ge=0, le=5000)
    total_extra_data_charges: Optional[float] = Field(default=None, ge=0, le=5000)


def check_consistency(c: CustomerInput) -> list[dict]:
    """Cross-field rules taken from the data (e.g. no phone service → no multiple lines).

    Fills values that are implied by another answer and returns a list of
    {"field", "message"} errors for answers that contradict each other.
    """
    errors = []
    if c.phone_service == "No":
        if c.multiple_lines == "Yes":
            errors.append({"field": "multiple_lines",
                           "message": "Must be 'No' when the customer has no phone service."})
        if c.avg_monthly_long_distance_charges not in (None, 0):
            errors.append({"field": "avg_monthly_long_distance_charges",
                           "message": "Must be 0 when the customer has no phone service."})
        c.multiple_lines = "No"
        c.avg_monthly_long_distance_charges = 0.0
    elif c.multiple_lines is None:
        errors.append({"field": "multiple_lines", "message": "Required when the customer has phone service."})

    if c.internet_type == NO_INTERNET:
        for f in INTERNET_ADDONS:
            if getattr(c, f) == "Yes":
                errors.append({"field": f, "message": "Must be 'No' when the customer has no internet service."})
            setattr(c, f, "No")
        for f in INTERNET_NUMERIC:
            if getattr(c, f) not in (None, 0):
                errors.append({"field": f, "message": "Must be 0 when the customer has no internet service."})
            setattr(c, f, 0.0)
    else:
        for f in INTERNET_ADDONS:
            if getattr(c, f) is None:
                errors.append({"field": f, "message": "Required when the customer has internet service."})
    return errors


# API field name → dataset column name used by the preprocessing pipeline
COLUMN_MAP = {
    "gender": "Gender", "age": "Age", "married": "Married", "number_of_dependents": "Number of Dependents",
    "tenure_in_months": "Tenure in Months", "contract": "Contract", "offer": "Offer",
    "number_of_referrals": "Number of Referrals", "cltv": "CLTV", "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines", "avg_monthly_long_distance_charges": "Avg Monthly Long Distance Charges",
    "internet_type": "Internet Type", "avg_monthly_gb_download": "Avg Monthly GB Download",
    "online_security": "Online Security", "online_backup": "Online Backup",
    "device_protection_plan": "Device Protection Plan", "premium_tech_support": "Premium Tech Support",
    "streaming_tv": "Streaming TV", "streaming_movies": "Streaming Movies", "streaming_music": "Streaming Music",
    "unlimited_data": "Unlimited Data", "monthly_charge": "Monthly Charge", "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method", "total_refunds": "Total Refunds",
    "total_extra_data_charges": "Total Extra Data Charges",
}

# Field metadata served to the frontend (GET /api/schema) so the form always matches the API
FIELD_SPECS = [
    # group, name, label, type, options / (min, max, step), required, help
    ("Customer profile", "gender", "Gender", "select", ["Female", "Male"], True, ""),
    ("Customer profile", "age", "Age", "number", (18, 100, 1), True, "Years"),
    ("Customer profile", "married", "Married", "select", ["Yes", "No"], True, ""),
    ("Customer profile", "number_of_dependents", "Number of dependents", "number", (0, 20, 1), True,
     "Children, parents or others living with the customer"),
    ("Account", "tenure_in_months", "Tenure (months)", "number", (1, 72, 1), True,
     "How long the customer has been with the company (1–72)"),
    ("Account", "contract", "Contract type", "select", CONTRACTS, True, ""),
    ("Account", "offer", "Last promotional offer accepted", "select", OFFERS, True, ""),
    ("Account", "number_of_referrals", "Number of referrals", "number", (0, 50, 1), True,
     "Friends or family the customer has referred"),
    ("Account", "cltv", "Customer lifetime value (CLTV)", "number", (0, 20000, 1), False,
     "Optional – company's value estimate (typically 2,000–6,500)"),
    ("Phone service", "phone_service", "Phone service", "select", ["Yes", "No"], True, ""),
    ("Phone service", "multiple_lines", "Multiple lines", "select", ["Yes", "No"], True, ""),
    ("Phone service", "avg_monthly_long_distance_charges", "Avg monthly long-distance charges ($)", "number",
     (0, 200, 0.01), False, "Optional"),
    ("Internet service", "internet_type", "Internet type", "select", INTERNET_TYPES, True, ""),
    ("Internet service", "avg_monthly_gb_download", "Avg monthly download (GB)", "number", (0, 500, 1), False,
     "Optional"),
    ("Internet service", "online_security", "Online security", "select", ["Yes", "No"], True, ""),
    ("Internet service", "online_backup", "Online backup", "select", ["Yes", "No"], True, ""),
    ("Internet service", "device_protection_plan", "Device protection plan", "select", ["Yes", "No"], True, ""),
    ("Internet service", "premium_tech_support", "Premium tech support", "select", ["Yes", "No"], True, ""),
    ("Internet service", "streaming_tv", "Streaming TV", "select", ["Yes", "No"], True, ""),
    ("Internet service", "streaming_movies", "Streaming movies", "select", ["Yes", "No"], True, ""),
    ("Internet service", "streaming_music", "Streaming music", "select", ["Yes", "No"], True, ""),
    ("Internet service", "unlimited_data", "Unlimited data", "select", ["Yes", "No"], True, ""),
    ("Billing", "monthly_charge", "Monthly charge ($)", "number", (0, 500, 0.01), True,
     "Current monthly bill for all services"),
    ("Billing", "paperless_billing", "Paperless billing", "select", ["Yes", "No"], True, ""),
    ("Billing", "payment_method", "Payment method", "select", PAYMENT_METHODS, True, ""),
    ("Billing", "total_refunds", "Total refunds ($)", "number", (0, 5000, 0.01), False, "Optional"),
    ("Billing", "total_extra_data_charges", "Total extra data charges ($)", "number", (0, 5000, 1), False,
     "Optional"),
]


def field_schema() -> list[dict]:
    out = []
    for group, name, label, ftype, spec, required, help_text in FIELD_SPECS:
        item = {"group": group, "name": name, "label": label, "type": ftype, "required": required,
                "help": help_text}
        if ftype == "select":
            item["options"] = spec
        else:
            item["min"], item["max"], item["step"] = spec
        if name in PHONE_DEPENDENT:
            item["depends_on"] = {"field": "phone_service", "disabled_when": "No",
                                  "value_when_disabled": "No" if ftype == "select" else 0}
        if name in INTERNET_ADDONS + INTERNET_NUMERIC:
            item["depends_on"] = {"field": "internet_type", "disabled_when": NO_INTERNET,
                                  "value_when_disabled": "No" if ftype == "select" else 0}
        out.append(item)
    return out


class Factor(BaseModel):
    feature: str
    value: str
    impact: float = Field(description="Contribution to the churn log-odds (positive = raises risk)")


class PredictionResult(BaseModel):
    churn_probability: float
    churn_probability_pct: str
    at_risk: bool
    prediction: str
    risk_level: Literal["High", "Moderate", "Low"]
    threshold: float
    factors_increasing_risk: list[Factor]
    factors_decreasing_risk: list[Factor]
    recommendations: list[str]
    warnings: list[str]


class BatchRequest(BaseModel):
    customers: list[CustomerInput] = Field(min_length=1, max_length=1000)


class BatchItem(PredictionResult):
    index: int
    customer_id: Optional[str] = None


class BatchResponse(BaseModel):
    count: int
    at_risk_count: int
    results: list[BatchItem]
    errors: list[dict] = []

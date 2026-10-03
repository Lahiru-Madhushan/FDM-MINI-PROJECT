# 📊 Telco Customer Churn Prediction

**IT3051 – Fundamentals of Data Mining | Mini Project 2026**

An end-to-end machine learning system for predicting whether a telecommunications customer is likely to **churn (leave)** or **stay** with the service provider.

---

## ⚡ Quick Start

| | |
| --- | --- |
| **What it does** | Predicts a customer's churn probability, explains the main reasons and suggests retention actions |
| **Final model** | Tuned **XGBoost** (decision threshold 0.51), chosen after comparing 4 algorithms |
| **Test performance** | F1 **0.721** · ROC-AUC **0.914** · Recall 0.762 · Precision 0.684 (1,409 held-out customers) |
| **Backend** | FastAPI service in `backend/` – loads `models/preprocessing_pipeline.pkl` and `models/xgboost_final.pkl` |
| **Frontend** | HTML / CSS / JavaScript in `frontend/` – served by the backend, so one command runs everything |
| **Requirements** | Python **3.11** and Git |

```powershell
# 1. Set up (once)
git clone https://github.com/Lahiru-Madhushan/FDM-MINI-PROJECT.git
cd FDM-MINI-PROJECT
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 2. Start the backend (it also serves the frontend)
.venv\Scripts\python -m uvicorn backend.app.main:app --reload
```

**3. Open the app:** <http://127.0.0.1:8000> – the frontend · <http://127.0.0.1:8000/docs> – interactive API documentation

> macOS / Linux: use `python3.11 -m venv .venv`, then `.venv/bin/python` instead of `.venv\Scripts\python`.
> Full details: [Getting Started](#-getting-started) · [Running the Backend](#-running-the-backend) · [Using the Frontend](#-using-the-frontend)

---

## 📖 Table of Contents

* [Quick Start](#-quick-start)

* [Overview](#-overview)
* [Problem Scenario](#-problem-scenario)
* [Project Objective](#-project-objective)
* [Stakeholders](#-stakeholders)
* [Dataset](#-dataset)
* [Machine Learning Task](#-machine-learning-task)
* [Project Workflow](#-project-workflow)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [Data Preprocessing](#-data-preprocessing)
* [Feature Engineering](#-feature-engineering)
* [Data Leakage Prevention](#-data-leakage-prevention)
* [Machine Learning Models](#-machine-learning-models)
* [Model Evaluation](#-model-evaluation)
* [Model Optimization](#-model-optimization)
* [System Architecture](#-system-architecture)
* [Technology Stack](#-technology-stack)
* [Project Structure](#-project-structure)
* [Getting Started](#-getting-started)
* [Running the Backend](#-running-the-backend)
* [Using the Frontend](#-using-the-frontend)
* [API Endpoints](#-api-endpoints)
* [Running the Tests](#-running-the-tests)
* [Deployment](#-deployment)
* [Troubleshooting](#-troubleshooting)
* [Team Contributions](#-team-contributions)
* [Evaluation 1](#-evaluation-1)
* [Future Development](#-future-development)
* [Project Scope](#-project-scope)
* [License](#-license)

---

# 🧭 Overview

Customer churn is an important business problem in the telecommunications industry. Customers may leave a provider because of factors such as contract type, service usage, pricing, customer experience, or other account-related conditions.

This project applies a complete **Data Mining and Machine Learning workflow** to predict whether a customer is likely to churn.

The system uses customer demographic, service, account, contract, billing, satisfaction, and customer-value information to generate a binary churn prediction.

The project follows the workflow required by the **IT3051 – Fundamentals of Data Mining Mini Project 2026**, progressing from problem understanding and dataset selection through EDA, preprocessing, feature engineering, model development, optimization, and an integrated prediction system.

---

# ❗ Problem Scenario

Telecommunications companies need to identify customers who may be at risk of leaving so that customer retention teams can investigate appropriate retention activities.

Traditional analysis may make it difficult to identify patterns across thousands of customer records.

The project addresses this problem by building a machine learning model that learns patterns from historical customer information and predicts whether a customer is likely to:

* **Stay**
* **Churn**

The prediction can support customer retention managers, marketing teams, and call center teams in identifying customers who may require further attention.

---

# 🎯 Project Objective

The main objective is to develop a **binary classification model** that predicts customer churn.

### Target Variable

**Churn Label**

| Value       | Meaning           |
| ----------- | ----------------- |
| `Yes` / `1` | Customer churned  |
| `No` / `0`  | Customer remained |

The model will use information available about a customer to predict the churn outcome.

---

# 👥 Stakeholders

Potential users of the system include:

* **Customer Retention Managers**

  * Identify customers with higher churn risk.
  * Support retention planning.

* **Marketing Teams**

  * Analyse customer segments and churn patterns.
  * Support targeted retention activities.

* **Call Center Agents**

  * Use customer information and predictions when interacting with customers.

The system is intended to provide prediction support rather than automatically make business or customer decisions.

---

# 📂 Dataset

## Dataset Name

**Telco Customer Churn (11.1.3+)**

## Original Source

IBM Cognos Analytics 11.1.3+ base samples dataset.

## Dataset Repository

The dataset was republished on Kaggle by **alfathterry**.

**Dataset URL:**

https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3

### Suggested Citation

IBM (2019). *Telco Customer Churn (11.1.3+)* [Data set]. Kaggle, republished by A. Terry. Original source: IBM Cognos Analytics 11.1.3+ base samples.

---

# 📌 Dataset Description

The dataset represents a fictional telecommunications company providing home phone and internet services to customers in California.

The supplied CSV contains:

* **7,043 customer records**
* **50 columns**

The dataset contains information related to:

* Customer demographics
* Geographic information
* Customer referrals
* Tenure
* Phone and internet services
* Service usage
* Contracts
* Billing
* Revenue
* Satisfaction
* Customer status
* Churn information
* Customer Lifetime Value (CLTV)

The dataset was selected because it provides a clear binary churn target and contains multiple customer, service, account, and billing attributes suitable for classification.

---

# 🤖 Machine Learning Task

### Task

**Binary Classification**

### Input

The model uses selected customer information such as:

* Demographic information
* Tenure
* Service subscriptions
* Contract information
* Payment method
* Billing information
* Satisfaction information
* Customer value information

### Output

A prediction indicating whether the customer is likely to:

```text
Churn
```

or

```text
Stay
```

---

# 🔄 Project Workflow

The project follows the complete data mining workflow required by the module:

```text
Problem Understanding
        ↓
Dataset Selection & Validation
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Data Leakage Detection
        ↓
Feature Selection
        ↓
Feature Engineering
        ↓
Encoding & Scaling
        ↓
Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Hyperparameter Optimization
        ↓
Final Model Selection
        ↓
Backend Development
        ↓
Frontend Development
        ↓
End-to-End Prediction System
```

The assignment requires at least **four suitable machine learning algorithms** to be implemented and compared.

---

# 📊 Exploratory Data Analysis

EDA was completed before the preprocessing stages.

Important observations from the dataset include:

### Dataset Size

```text
7,043 rows × 50 columns
```

### Target Distribution

| Churn | Customers | Percentage |
| ----- | --------: | ---------: |
| No    |     5,174 |     73.46% |
| Yes   |     1,869 |     26.54% |

This indicates class imbalance, so stratified train/test splitting and suitable evaluation metrics will be considered.

### Duplicate Records

```text
0 duplicate rows
```

### Missing Values

Important missing-value observations include:

* `Offer` → 3,877 missing
* `Internet Type` → 1,526 missing
* `Churn Category` → 5,174 missing
* `Churn Reason` → 5,174 missing

Missing values will be interpreted based on their meaning rather than automatically deleting the affected records.

### Financial Validity

The checked financial variables contained:

```text
0 negative values
```

### Outliers

IQR-based analysis identified potential outliers in variables such as:

* Number of Dependents
* Number of Referrals
* Total Refunds
* Total Extra Data Charges
* Total Long Distance Charges
* Population

These observations will not automatically be removed because some may represent legitimate customer behaviour.

---

# 🧹 Data Preprocessing

The preprocessing stage is divided among the three team members.

## Member 1 – Data Cleaning & Data Quality

Responsibilities include:

* Data-type verification
* Missing-value analysis
* Duplicate checking
* Invalid-value checking
* Outlier investigation
* Data-quality documentation

The goal is to produce a clean and internally consistent dataset.

---

## Member 2 – Data Leakage, Feature Selection & Feature Engineering

Responsibilities include:

* Identifying target leakage
* Removing identifiers
* Selecting legitimate predictors
* Reviewing redundant features
* Creating justified derived features
* Documenting feature-selection decisions

---

## Member 3 – Encoding, Scaling & Train/Test Pipeline

Responsibilities include:

* Separating target and predictors
* Train/test splitting
* Stratified sampling
* Categorical encoding
* Numerical imputation
* Feature scaling
* Building the `ColumnTransformer` pipeline
* Preventing preprocessing leakage

---

# ⚠️ Data Leakage Prevention

Data leakage is an important consideration in this project.

The following variables are excluded from predictive inputs:

| Feature           | Decision | Reason                                |
| ----------------- | -------- | ------------------------------------- |
| `Churn Label`     | Target   | Prediction target                     |
| `Customer ID`     | Remove   | Unique identifier                     |
| `Churn Score`     | Remove   | Contains churn-related information    |
| `Churn Category`  | Remove   | Describes the churn outcome           |
| `Churn Reason`    | Remove   | Describes why a customer churned      |
| `Customer Status` | Remove   | Directly corresponds to churn outcome |

The key principle is:

> A feature should only be used if the information would be available at the time the churn prediction is being made.

Post-outcome information must not be used as model input.

---

# 🛠️ Feature Engineering

Feature engineering will be limited to features that have a clear business or modelling justification.

### Proposed Features

#### Total Services

Counts selected subscribed services to represent the breadth of services used by a customer.

```text
Total Services =
number of selected active service indicators
```

#### Tenure Group

Customers can be grouped into interpretable tenure ranges.

This may help represent different stages of the customer lifecycle if supported by model experiments.

Feature engineering will be evaluated experimentally rather than assuming that every engineered feature improves performance.

---

# 🔤 Encoding

Categorical features cannot be directly supplied to most numerical machine learning algorithms.

The preprocessing pipeline uses **One-Hot Encoding** for categorical variables.

Example:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

`handle_unknown="ignore"` allows the pipeline to handle categories that were not present during training.

---

# 📏 Numerical Scaling

Numerical variables may have different ranges.

For example:

```text
Age
Monthly Charge
Total Charges
CLTV
Population
```

The preprocessing pipeline can use:

```python
StandardScaler()
```

Scaling is particularly relevant for scale-sensitive algorithms such as Logistic Regression.

Tree-based models generally do not require feature scaling, but keeping preprocessing inside a controlled pipeline allows consistent model integration.

---

# 🔀 Train/Test Split

The dataset is divided into training and testing data using a stratified split.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Configuration

```text
Training data: 80%
Testing data: 20%
Random state: 42
Stratification: Enabled
```

Stratification helps maintain approximately the same churn/non-churn class proportion in both datasets.

The test set remains unseen while preprocessing parameters are learned.

---

# 🧩 Preprocessing Pipeline

The project uses a `ColumnTransformer` to apply different preprocessing operations to numerical and categorical features.

```text
                    Selected Features
                           │
              ┌────────────┴────────────┐
              │                         │
        Numerical Features        Categorical Features
              │                         │
       Median Imputation          Missing-value Handling
              │                         │
       Standard Scaling           One-Hot Encoding
              │                         │
              └────────────┬────────────┘
                           │
                    ML-ready Features
```

The preprocessing pipeline is fitted using the training data and then applied to both training and testing data.

This helps prevent information from the test set influencing preprocessing decisions.

---

# 🤖 Machine Learning Models

The project plans to implement and compare multiple classification algorithms.

### Planned Algorithms

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting / XGBoost**

The models will be compared systematically using appropriate evaluation metrics.

The final model will be selected after model comparison and optimization rather than selecting a model in advance.

---

# 📈 Model Evaluation

Because the dataset contains an imbalanced target distribution, accuracy alone will not be sufficient.

Potential evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

The selected metrics will be used to compare the models and understand different types of prediction errors.

---

# ⚙️ Model Optimization

After baseline model development, suitable models will undergo hyperparameter tuning.

The optimization stage will investigate:

* Hyperparameter combinations
* Feature-selection effects
* Feature-engineering effects
* Baseline vs tuned performance
* Validation performance

The final model will be selected based on the experimental results and documented justification.

---

# 🏗️ System Architecture

The planned system will connect the trained machine learning model to a backend and frontend.

```text
┌──────────────────────┐
│        USER          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Frontend        │
│ Customer Input Form  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│       Backend        │
│  Input Validation    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Preprocessing        │
│ Pipeline             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Final ML Model     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Churn Prediction     │
│  Stay / Churn        │
└──────────────────────┘
```

The final backend must load the trained model and required preprocessing pipeline, validate inputs, apply the same preprocessing used during training, and return the prediction.

---

# 🧰 Technology Stack

| Layer                | Technology                                                                     |
| -------------------- | ------------------------------------------------------------------------------ |
| Programming Language | Python                                                                         |
| Data Processing      | Pandas, NumPy                                                                  |
| Visualization        | Matplotlib / Seaborn                                                           |
| Machine Learning     | Scikit-learn                                                                   |
| Models               | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting / XGBoost |
| Preprocessing        | Scikit-learn Pipeline / ColumnTransformer                                      |
| Model Serialization  | Joblib / Pickle                                                                |
| Backend              | Python-based API                                                               |
| Frontend             | Web-based UI                                                                   |
| Development          | Jupyter Notebook / VS Code                                                     |
| Version Control      | Git / GitHub                                                                   |

---

# 📁 Project Structure

```text
FDM-MINI-PROJECT/
│
├── Dataset/
│   └── telco.csv                         # raw IBM Telco dataset (7,043 × 50)
│
├── Notebook/
│   ├── Preprocessing/
│   │   ├── EDA.ipynb                     # data understanding, statistics, leakage discovery
│   │   ├── Data_Cleaning.ipynb           # → cleaned_data.csv
│   │   ├── Feature_Selection.ipynb       # → feature_selected_data.csv (29 features)
│   │   ├── Data_Preprocessing.ipynb      # → train/test_processed.csv + preprocessing pipeline
│   │   └── Preprocessing_Full_Flow.ipynb # whole preprocessing flow in one notebook → SplitData/
│   └── modelCreation/
│       ├── LogisticRegression/           # Logistic Regression
│       ├── DecissionTree/                # Decision Tree
│       ├── RandomForest/                 # Random Forest
│       ├── XGBoost/                      # XGBoost (final model)
│       └── ModelComparison/              # evaluation of all 4 models + figures/ for the report
│
├── SplitData/                            # X/y train and test splits
├── train_processed.csv, test_processed.csv
│
├── models/
│   ├── preprocessing_pipeline.pkl        # fitted ColumnTransformer (imputers, scaler, one-hot)
│   ├── xgboost_final.pkl                 # final model used by the backend
│   ├── xgboost_final_metadata.json       # decision threshold, feature order, test metrics
│   ├── decision_tree_final.pkl
│   └── random_forest_final.pkl
│
├── backend/
│   ├── app/
│   │   ├── main.py                       # FastAPI app and API endpoints
│   │   ├── predictor.py                  # loads the model; preprocessing, predictions, explanations
│   │   ├── schemas.py                    # input fields, validation rules, response format
│   │   └── config.py                     # paths and limits
│   ├── tests/test_api.py                 # API tests
│   └── requirements.txt                  # app dependencies
│
├── frontend/
│   ├── index.html, styles.css, app.js    # Churn Risk Checker web interface
│   ├── examples.json                     # example customers (high / moderate / low risk)
│   └── sample_customers.csv              # sample file for CSV scoring
│
├── requirements.txt                      # all dependencies (app + notebooks)
├── render.yaml                           # deployment configuration for Render
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

* **Python 3.11** – the models were trained with Python 3.11 and scikit-learn 1.9.1; using the same versions guarantees the saved models load correctly
* **Git**
* VS Code or Jupyter (only needed to open the notebooks)

## 1. Clone the Repository

```bash
git clone https://github.com/Lahiru-Madhushan/FDM-MINI-PROJECT.git
cd FDM-MINI-PROJECT
```

## 2. Create a Virtual Environment

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call `.venv\Scripts\python` directly as in the commands below.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
# everything: the app and the notebooks
python -m pip install -r requirements.txt

# or the app only (backend + frontend)
python -m pip install -r backend/requirements.txt
```

## 4. Check the Installation

```bash
python -m pytest backend/tests -q
```

All tests should pass (see [Running the Tests](#-running-the-tests)).

## 5. Run the Notebooks (optional)

The trained models are already saved in `models/`, so the notebooks are only needed to reproduce or inspect the work. Open them in VS Code or Jupyter, select the `.venv` kernel, and run them in this order:

| Step | Notebook | Output |
| --- | --- | --- |
| 1 | `Notebook/Preprocessing/EDA.ipynb` | Data understanding and charts |
| 2 | `Notebook/Preprocessing/Data_Cleaning.ipynb` | `cleaned_data.csv` |
| 3 | `Notebook/Preprocessing/Feature_Selection.ipynb` | `feature_selected_data.csv` |
| 4 | `Notebook/Preprocessing/Data_Preprocessing.ipynb` | `train_processed.csv`, `test_processed.csv`, `models/preprocessing_pipeline.pkl` |
| 5 | `Notebook/modelCreation/<Model>/…ipynb` | One notebook per algorithm; final models saved in `models/` |
| 6 | `Notebook/modelCreation/ModelComparison/Model_Evaluation_Comparison.ipynb` | Comparison of all four models; charts and tables in `figures/` |

`Notebook/Preprocessing/Preprocessing_Full_Flow.ipynb` runs steps 2–4 in a single notebook.

---

# 🔌 Running the Backend

The backend is a **FastAPI** service. When it starts it loads the saved preprocessing pipeline, the final XGBoost model and its decision threshold from `models/`. It also serves the frontend, so this one command starts the whole system.

From the project root:

```powershell
# Windows
.venv\Scripts\python -m uvicorn backend.app.main:app --reload
```

```bash
# macOS / Linux
.venv/bin/python -m uvicorn backend.app.main:app --reload
```

When it is ready you will see:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

| Option | Meaning |
| --- | --- |
| `--reload` | Restarts automatically when you change backend code (for development) |
| `--port 8001` | Use another port if 8000 is busy (then open `http://127.0.0.1:8001`) |
| `--host 0.0.0.0` | Allow other devices on your network to open the app |

**Check it is running:** open <http://127.0.0.1:8000/api/health> – it should show `"status": "ok"` and the model details.

**Stop it:** press `Ctrl + C` in the terminal.

---

# 💻 Using the Frontend

1. Start the backend (see above) and keep the terminal open.
2. Open **<http://127.0.0.1:8000>** in a browser.
3. Check that the badge at the top right says **Service online**.

### Single customer

1. Fill in the customer's details – fields marked **\*** are required. Optional numbers (e.g. CLTV) can be left blank; the system then uses a typical value and tells you so.
2. Or click **High risk**, **Moderate risk** or **Low risk** to load a real example customer.
3. Click **Predict churn risk**.

Fields that do not apply are filled in automatically – for example, choosing *No Internet Service* switches off the internet add-ons. Invalid values (e.g. age 150) are highlighted with a message before anything is sent.

The result panel shows:

| Part | Meaning |
| --- | --- |
| **Churn probability** | Chance that the customer will leave (0–100%) |
| **Decision** | *Likely to churn* when the probability is **51% or higher** (the model's decision threshold) |
| **Risk level** | **High** ≥ 51% (flagged) · **Moderate** 30–51% (keep an eye on) · **Low** < 30% |
| **What raises / lowers the risk** | The customer details that pushed this prediction up or down the most |
| **Suggested actions** | Retention actions linked to the main risk factors |
| **Warnings** | E.g. a value was estimated, or is outside the range seen in training |

### Many customers (CSV)

1. Open the **Many customers (CSV)** tab.
2. Click **Download sample CSV** to see the expected format, or use your own file. Columns can use the original dataset names (e.g. `Monthly Charge`) or the API names (e.g. `monthly_charge`); a `Customer ID` column is kept as an identifier. Up to 5,000 rows.
3. Choose the file and click **Score file**.
4. Customers are listed from highest to lowest risk, with the main risk factor and a suggested action. Tick **Show only customers likely to churn** to filter, and click **Download results (CSV)** to export the list. Rows with errors are listed separately with the reason.

> The page can also be opened directly from `frontend/index.html` or through VS Code Live Server – it then connects to the backend at `http://127.0.0.1:8000` automatically. To use a backend somewhere else, add `?api=<backend-url>` to the page address.

---

# 🔗 API Endpoints

Interactive documentation with a **Try it out** button for every endpoint: <http://127.0.0.1:8000/docs>

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service status, model version, decision threshold and test metrics |
| `GET` | `/api/schema` | Input fields, allowed values and ranges (used to build the form) |
| `POST` | `/api/predict` | Predict one customer |
| `POST` | `/api/predict/batch` | Predict up to 1,000 customers (JSON) |
| `POST` | `/api/predict/csv` | Score an uploaded CSV file (up to 5,000 rows) |

Example request (PowerShell):

```powershell
$customer = (Get-Content frontend\examples.json | ConvertFrom-Json).high
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/predict -Method Post `
  -ContentType "application/json" -Body ($customer | ConvertTo-Json)
```

Example response (shortened):

```json
{
  "churn_probability": 0.951,
  "churn_probability_pct": "95.1%",
  "at_risk": true,
  "prediction": "Likely to churn",
  "risk_level": "High",
  "threshold": 0.51,
  "factors_increasing_risk": [{ "feature": "Contract", "value": "Month-to-Month", "impact": 1.11 }],
  "recommendations": ["Offer an incentive (e.g. a discount) to move from a month-to-month to a one- or two-year contract."],
  "warnings": []
}
```

Invalid or inconsistent inputs return **HTTP 422** with one message per field:

```json
{
  "detail": "Invalid input",
  "errors": [{ "field": "age", "message": "Input should be greater than or equal to 18" }]
}
```

---

# 🧪 Running the Tests

```bash
python -m pytest backend/tests -q
```

The tests cover valid predictions; missing, invalid and inconsistent inputs; batch and CSV scoring; and an end-to-end check that the API reproduces the notebooks' predictions for all 1,409 test customers (F1 0.7206).

---

# 🌐 Deployment

The app is deployed as a single web service on **[Render](https://render.com)** (free plan), using the configuration in `render.yaml`.

1. Sign in to Render with GitHub and give it access to this repository.
2. Click **New → Blueprint**, select the repository, then **Apply**.
3. After the build (about 3–5 minutes) the app is available at the URL Render shows, e.g. `https://churn-risk-checker.onrender.com`.

Without the blueprint, create a **Web Service** with build command `pip install -r backend/requirements.txt`, start command `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`, health check path `/api/health` and environment variable `PYTHON_VERSION=3.11.3`.

Every push to `main` redeploys automatically. On the free plan the app sleeps after 15 minutes without visitors, and the first visit then takes about a minute – open it shortly before a demo.

---

# 🩺 Troubleshooting

| Problem | Solution |
| --- | --- |
| Page shows **Service offline** | The backend is not running – start it, then reload the page (`Ctrl + F5`). Open the app at `http://127.0.0.1:8000`. |
| `ModuleNotFoundError` (e.g. `fastapi`, `xgboost`) | Use the virtual environment's Python (`.venv\Scripts\python …`) and install the requirements. |
| Port 8000 already in use | Start with `--port 8001` and open `http://127.0.0.1:8001`. |
| `Activate.ps1` cannot be loaded | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call `.venv\Scripts\python` directly. |
| Warnings when loading the models | Install the exact versions from `requirements.txt` (scikit-learn 1.9.1, xgboost 3.2.0). |
| Frontend changes do not appear | Hard-refresh the browser with `Ctrl + F5`. |

---

# 👥 Team Contributions

The preprocessing stage is divided into three connected pipelines.

| Member   | Main Responsibility                              | Key Deliverable                 |
| -------- | ------------------------------------------------ | ------------------------------- |
| Member 1 | Data Cleaning & Data Quality                     | Cleaned dataset                 |
| Member 2 | Leakage, Feature Selection & Feature Engineering | Final feature set               |
| Member 3 | Encoding, Scaling & Train/Test Pipeline          | ML-ready preprocessing pipeline |

Although responsibilities are divided, every team member is expected to understand the complete project and be able to explain their own technical decisions during individual evaluations.

---

# 📝 Evaluation 1

**Progress Evaluation 1 – 30%**

Evaluation 1 is an individual viva covering the project work completed up to the preprocessing stage.

Topics include:

* Problem understanding
* Dataset selection
* Dataset characteristics
* Target variable
* EDA findings
* Data-quality issues
* Preprocessing techniques
* Feature engineering
* Feature selection
* Data leakage
* Leakage prevention
* Individual contribution

The assignment identifies Progress Evaluation 1 as an individual evaluation worth **30%**.

---

# 📌 Evaluation 1 Preprocessing Pipeline

```text
Raw Telco Dataset
       │
       ▼
┌─────────────────────┐
│ Member 1            │
│ Data Cleaning       │
│ & Data Quality      │
└──────────┬──────────┘
           │
           ▼
     Cleaned Dataset
           │
           ▼
┌─────────────────────┐
│ Member 2            │
│ Leakage Removal     │
│ Feature Selection   │
│ Feature Engineering │
└──────────┬──────────┘
           │
           ▼
    Final Feature Set
           │
           ▼
┌─────────────────────┐
│ Member 3            │
│ Encoding            │
│ Scaling             │
│ Train/Test Pipeline │
└──────────┬──────────┘
           │
           ▼
      ML-ready Data
           │
           ▼
    Model Development
```

---

# 🗺️ Project Roadmap

| Stage                     | Status          |
| ------------------------- | --------------- |
| Problem Definition        | ✅ Completed     |
| Dataset Selection         | ✅ Completed     |
| Dataset Validation        | ✅ Completed     |
| EDA                       | ✅ Completed     |
| Data Cleaning             | 🔄 Evaluation 1 |
| Feature Selection         | 🔄 Evaluation 1 |
| Feature Engineering       | 🔄 Evaluation 1 |
| Encoding & Scaling        | 🔄 Evaluation 1 |
| Train/Test Pipeline       | 🔄 Evaluation 1 |
| Model Development         | ⏳ Planned       |
| Model Comparison          | ⏳ Planned       |
| Hyperparameter Tuning     | ⏳ Planned       |
| Final Model Selection     | ⏳ Planned       |
| Backend Development       | ⏳ Planned       |
| Frontend Development      | ⏳ Planned       |
| System Testing            | ⏳ Planned       |
| Technical Report          | ⏳ Planned       |
| Final Presentation & Demo | ⏳ Planned       |

---

# 🎯 Project Scope

## In Scope

* Customer churn prediction
* Exploratory data analysis
* Data cleaning
* Missing-value handling
* Duplicate and validity checks
* Outlier investigation
* Data leakage prevention
* Feature selection
* Feature engineering
* Categorical encoding
* Numerical scaling
* Train/test preprocessing
* Multiple machine learning models
* Model comparison
* Hyperparameter optimization
* Final model selection
* Backend prediction service
* Frontend prediction interface
* System testing
* Technical documentation

## Out of Scope

The project does not attempt to:

* Automatically contact customers
* Automatically provide discounts or retention offers
* Replace human customer-retention decisions
* Perform real-time telecom network monitoring
* Manage telecom infrastructure
* Perform network intrusion detection

The system is focused specifically on **customer churn prediction**.

---

# ⚠️ Dataset Limitations

The dataset represents a fictional telecommunications provider and a specific customer population.

Important limitations include:

* The dataset represents customers in California.
* It represents a single-quarter snapshot.
* Patterns may not generalize to other telecom providers, regions, or time periods.
* Some geographic variables may have high dimensionality.
* Churn-related fields can introduce target leakage if incorrectly included.
* Class imbalance must be considered during modelling and evaluation.

---

# 🔐 Ethical Considerations

Churn predictions may influence which customers receive additional attention from a business.

Therefore, the project should consider:

* Fairness across demographic groups
* Responsible use of predictions
* Avoiding discriminatory treatment
* Appropriate handling of customer-level information
* Transparency regarding model predictions
* Human involvement in business decisions

The model should be treated as a decision-support tool rather than an automatic decision-maker.

---

# 📚 Project Requirements

The project follows the requirements of the **IT3051 – Fundamentals of Data Mining Mini Project 2026**.

The assignment requires the project to progress from problem understanding and dataset selection through data preparation, EDA, modelling, optimization, and implementation of a user-friendly prediction system.

The final project includes:

* Approved dataset and source
* Preprocessing source code
* Model development
* Trained final model
* Integrated prediction system
* Technical report
* Presentation materials
* Experiment and evaluation evidence

---

# 📄 Academic Context

**Module:** IT3051 – Fundamentals of Data Mining
**Project:** Mini Project 2026
**Problem:** Telco Customer Churn Prediction
**Task:** Binary Classification
**Dataset:** Telco Customer Churn (11.1.3+)
**Target:** Churn Label

---

# 👨‍💻 Project Status

This repository is being developed as part of the **IT3051 – Fundamentals of Data Mining Mini Project 2026**.

The current development stage focuses on:

```text
EDA
 ↓
Data Cleaning
 ↓
Feature Selection & Engineering
 ↓
Encoding & Scaling
 ↓
Train/Test Preprocessing
```

The next stages will cover machine learning model development, comparison, optimization, and integration into a functional prediction system.

---

# 📜 License

This project is developed for academic purposes as part of:

**IT3051 – Fundamentals of Data Mining**

Dataset ownership and usage are subject to the original dataset source and its applicable terms.

---

## 👥 Built by the FDM Mini Project Team

**Telco Customer Churn Prediction – 2026**

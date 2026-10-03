# 📊 Telco Customer Churn Prediction

**IT3051 – Fundamentals of Data Mining | Mini Project 2026**

An end-to-end machine learning system for predicting whether a telecommunications customer is likely to **churn (leave)** or **stay** with the service provider.

---

## 📖 Table of Contents

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
├── data/
│   ├── telco.csv
│   └── cleaned_data.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Cleaning_Member1.ipynb
│   ├── 03_Feature_Selection_Member2.ipynb
│   ├── 04_Preprocessing_Member3.ipynb
│   └── 05_Model_Development.ipynb
│
├── src/
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   └── model_optimization.py
│
├── models/
│   ├── preprocessing_pipeline.pkl
│   └── final_model.pkl
│
├── backend/
│   └── ...
│
├── frontend/
│   └── ...
│
├── docs/
│   ├── Evaluation_1_Preprocessing_Document.docx
│   └── technical_report.docx
│
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

Install:

* Python 3.10+
* Jupyter Notebook or VS Code
* Git

## 1. Clone the Repository

```bash
git clone <repository-url>
cd FDM-MINI-PROJECT
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the EDA Notebook

```text
notebooks/01_EDA.ipynb
```

## 5. Run Preprocessing

Execute the preprocessing notebooks in the required sequence:

```text
02_Data_Cleaning_Member1.ipynb
03_Feature_Selection_Member2.ipynb
04_Preprocessing_Member3.ipynb
```

## 6. Model Development

After preprocessing:

```text
05_Model_Development.ipynb
```

will be used for model training, comparison and optimization.

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

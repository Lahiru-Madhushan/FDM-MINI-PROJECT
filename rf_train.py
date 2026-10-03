import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X_train = pd.read_csv('SplitData/X_train.csv')
X_test = pd.read_csv('SplitData/X_test.csv')
y_train = pd.read_csv('SplitData/y_train.csv').iloc[:, 0]
y_test = pd.read_csv('SplitData/y_test.csv').iloc[:, 0]

print('=== DATA INSPECTION ===')
print('X_train shape:', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape :', y_test.shape)
print('y_train distribution:')
print(y_train.value_counts().sort_index().to_string())
print('y_test distribution:')
print(y_test.value_counts().sort_index().to_string())

baseline = RandomForestClassifier(random_state=42, n_estimators=200)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

print('\n=== BASELINE RANDOM FOREST RESULTS ===')
base_accuracy = accuracy_score(y_test, y_pred_base)
base_precision = precision_score(y_test, y_pred_base, zero_division=0)
base_recall = recall_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base)
base_cm = confusion_matrix(y_test, y_pred_base)
print('Accuracy :', round(base_accuracy, 4))
print('Precision:', round(base_precision, 4))
print('Recall   :', round(base_recall, 4))
print('F1-score :', round(base_f1, 4))
print('Confusion Matrix:')
print(base_cm)
print('Classification Report:')
print(classification_report(y_test, y_pred_base, target_names=['No Churn', 'Churn']))

param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('\n=== HYPERPARAMETER TUNING ===')
print('Scoring metric: F1-score on the positive class (churn), because the churn rate is lower than the non-churn rate and F1 balances precision and recall for the minority class.')
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train)
print('Best hyperparameters:', grid.best_params_)
print('Best cross-validation score:', round(grid.best_score_, 4))

tuned = grid.best_estimator_
y_pred_tuned = tuned.predict(X_test)
print('\n=== TUNED RANDOM FOREST RESULTS ===')
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_precision = precision_score(y_test, y_pred_tuned, zero_division=0)
tuned_recall = recall_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned)
tuned_cm = confusion_matrix(y_test, y_pred_tuned)
print('Accuracy :', round(tuned_accuracy, 4))
print('Precision:', round(tuned_precision, 4))
print('Recall   :', round(tuned_recall, 4))
print('F1-score :', round(tuned_f1, 4))
print('Confusion Matrix:')
print(tuned_cm)
print('Classification Report:')
print(classification_report(y_test, y_pred_tuned, target_names=['No Churn', 'Churn']))

comparison = pd.DataFrame(
    {
        'Accuracy': [base_accuracy, tuned_accuracy],
        'Precision': [base_precision, tuned_precision],
        'Recall': [base_recall, tuned_recall],
        'F1-score': [base_f1, tuned_f1],
    },
    index=['Baseline Random Forest', 'Tuned Random Forest'],
)
print('\n=== BASELINE VS TUNED COMPARISON ===')
print(comparison.round(4).to_string())

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': tuned.feature_importances_,
}).sort_values('Importance', ascending=False)
print('\n=== FEATURE IMPORTANCE ===')
print(feature_importance.head(10).to_string(index=False))

joblib.dump(tuned, 'models/random_forest_final.pkl')
loaded_model = joblib.load('models/random_forest_final.pkl')
print('\n=== SAVED MODEL VERIFICATION ===')
print('Saved model loaded successfully.')
print('Sample predictions from loaded model:', loaded_model.predict(X_test.head(3)).tolist())

print('\n=== SUMMARY ===')
print('Baseline accuracy:', round(base_accuracy, 4))
print('Tuned accuracy  :', round(tuned_accuracy, 4))
print('Best params     :', grid.best_params_)
print('Best CV F1      :', round(grid.best_score_, 4))

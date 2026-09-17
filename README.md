# ML_case_study1
# Hospital Readmission Prediction

Predicts 30-day hospital readmission using patient records (vitals, BMI, blood pressure, medications, discharge info) with L2-regularized Logistic Regression.

## Highlights
- **Dataset**: 30,000 patient records
- **Preprocessing**: Split blood pressure into systolic/diastolic, one-hot encoded categorical features, scaled numeric features
- **Model**: Logistic Regression (L2, C=1.0)

## Results
**ROC-AUC:** 0.564

| | Pred: No | Pred: Yes |
|---|---|---|
| **Actual: No** | 5265 | 0 |
| **Actual: Yes** | 735 | 0 |

## FN vs FP
- **False Negative** (missed readmission): more costly — patient misses follow-up care, risking a preventable readmission.
- **False Positive** (false alarm): low cost — just an extra check-in call.

Model currently predicts "No" for everyone due to class imbalance — next step: use `class_weight='balanced'` and a lower decision threshold.

## Tech
Python, pandas, scikit-learn, matplotlib

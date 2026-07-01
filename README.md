# DevelopersHub Data Science & Analytics Internship — Phase 2

## Task 1: Term Deposit Subscription Prediction

### Objective
Predict whether a bank customer will subscribe to a term deposit based on 
marketing campaign data using Machine Learning and Explainable AI (SHAP).

### Dataset
- **Source:** Kaggle — Bank Marketing Dataset (UCI)
- **Rows:** 41,188 | **Columns:** 21
- **Target:** y (yes = subscribed, no = not subscribed)

### Approach
- Explored dataset and checked class imbalance (88% No, 11% Yes)
- Encoded 11 categorical features using Label Encoding
- Performed EDA with count plot, box plots and correlation heatmap
- Trained Random Forest Classifier (80/20 split)
- Evaluated using Accuracy, F1 Score, Confusion Matrix and ROC Curve
- Applied SHAP to explain overall and individual predictions

### Results
| Metric | Value |
|--------|-------|
| Accuracy | 91.33% |
| F1 Score | 0.57 |
| AUC Score | 0.94 |

### Key Insights
- Duration (call length) is the strongest predictor of subscription
- Dataset is highly imbalanced — F1 Score is more reliable than Accuracy
- Economic indicators (emp.var.rate, euribor3m) significantly influence decisions
- SHAP waterfall plots explain individual customer predictions clearly

### Libraries Used
`pandas` `numpy` `matplotlib` `seaborn` `scikit-learn` `shap`

---
*DevelopersHub Corporation — Data Science & Analytics Internship Phase 2*

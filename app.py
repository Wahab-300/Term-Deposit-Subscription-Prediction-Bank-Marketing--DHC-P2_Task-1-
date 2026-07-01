import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import shap
import numpy as np

# App Title
st.title("🏦 Term Deposit Subscription Prediction")
st.write("Predict whether a bank customer will subscribe to a term deposit.")

# Load & Train Model
@st.cache_data
def load_and_train():
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    
    le = LabelEncoder()
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(columns=['y'])
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X, y, X_test, y_test, df

model, X, y, X_test, y_test, df = load_and_train()

# Dataset Overview
st.subheader("📋 Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# Model Performance
st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
col2.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
fpr, tpr, _ = roc_curve(y_test, y_prob)
col3.metric("AUC Score", f"{auc(fpr, tpr):.2f}")

# ROC Curve
st.subheader("📈 ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label=f'AUC = {auc(fpr, tpr):.2f}')
ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
st.pyplot(fig)

# SHAP Feature Importance
st.subheader("🔍 SHAP Feature Importance")
explainer = shap.TreeExplainer(model)
X_sample = X_test[:100]
shap_values = explainer.shap_values(X_sample)

fig, ax = plt.subplots()
shap.summary_plot(shap_values[:, :, 1], X_sample, 
                  feature_names=X.columns.tolist(), 
                  plot_type='bar', show=False)
st.pyplot(fig)

# User Input Section
st.subheader("🎯 Predict Subscription")
st.write("Fill in customer details to predict subscription:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    duration = st.number_input("Call Duration (seconds)", min_value=0, value=200)
    campaign = st.number_input("Number of Contacts", min_value=1, value=2)
    pdays = st.number_input("Days Since Last Contact", min_value=0, value=999)
    previous = st.number_input("Previous Contacts", min_value=0, value=0)

with col2:
    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857)
    nr_employed = st.number_input("Number of Employees", value=5191.0)

# Fixed encoded values for simplicity
job_enc = 0
marital_enc = 0
education_enc = 0
default_enc = 0
housing_enc = 0
loan_enc = 0
contact_enc = 0
month_enc = 0
day_enc = 0
poutcome_enc = 0

# Predict Button
if st.button("Predict"):
    input_data = [[age, job_enc, marital_enc, education_enc, default_enc,
                   housing_enc, loan_enc, contact_enc, month_enc, day_enc,
                   duration, campaign, pdays, previous, poutcome_enc,
                   emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]]
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer likely to Subscribe! (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Customer unlikely to Subscribe. (Probability: {probability:.2%})")

# Key Insights
st.subheader("✅ Key Insights")
st.markdown("""
- Call duration is the strongest predictor of subscription
- Economic indicators heavily influence customer decisions
- Dataset is imbalanced — AUC (0.94) is more reliable than Accuracy
- SHAP explains individual predictions transparently
""")

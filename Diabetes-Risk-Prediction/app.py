
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# Page settings
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

# Load saved files
@st.cache_resource
def load_files():
    model = XGBClassifier()
    model.load_model(str(BASE / "diabetes_model.json"))

    scaler = joblib.load(BASE / "scaler.pkl")
    features = list(joblib.load(BASE / "feature_columns.pkl"))

    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv(BASE / "diabetes_prediction_dataset.csv")

try:
    model, scaler, features = load_files()
    df = load_data()
except Exception as e:
    st.error(f"File loading error: {e}")
    st.stop()

# Prepare data for evaluation
# This assumes training used pd.get_dummies and a scaler.
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

X = pd.get_dummies(X)
X = X.reindex(columns=features, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

try:
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
except Exception as e:
    st.error(f"Could not calculate metrics: {e}")
    st.stop()

# Sidebar
st.sidebar.title("🩺 Diabetes AI")
page = st.sidebar.radio(
    "Choose Page",
    ["Dashboard", "Risk Prediction", "Dataset Explorer", "About Model"]
)

# Header
st.title("Diabetes Risk Prediction System")
st.write("Machine Learning Dashboard using XGBoost")
st.divider()

# Dashboard
if page == "Dashboard":

    st.subheader("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Diabetes Cases", int(y.sum()))
    c3.metric("No Diabetes", int((y == 0).sum()))
    c4.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("🎯 Model Performance")

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{accuracy * 100:.2f}%")
    b.metric("Precision", f"{precision * 100:.2f}%")
    c.metric("Recall", f"{recall * 100:.2f}%")
    d.metric("F1 Score", f"{f1 * 100:.2f}%")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score (%)": [
            accuracy * 100,
            precision * 100,
            recall * 100,
            f1 * 100
        ]
    }).set_index("Metric")

    st.subheader("Performance Graph")
    st.bar_chart(metrics_df)

    left, right = st.columns(2)

    with left:
        st.subheader("Diabetes Distribution")
        st.bar_chart(df["diabetes"].value_counts().sort_index())

    with right:
        st.subheader("Age Distribution")
        st.bar_chart(df["age"].value_counts(bins=10).sort_index())

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_yticklabels(["No Diabetes", "Diabetes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Metrics are calculated on a held-out split of the dataset.")

# Risk Prediction
elif page == "Risk Prediction":

    st.subheader("🧑 Patient Information")

    with st.form("patient_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            age = st.slider("Age", 1, 100, 35)
            hypertension = st.selectbox("Hypertension", [0, 1])
            heart_disease = st.selectbox("Heart Disease", [0, 1])

        with col2:
            smoking = st.selectbox(
                "Smoking History",
                ["never", "No Info", "current", "former", "ever", "not current"]
            )
            bmi = st.number_input("BMI", 10.0, 80.0, 25.0)
            hba1c = st.number_input("HbA1c Level", 3.0, 20.0, 5.5)
            glucose = st.number_input("Blood Glucose", 50, 500, 100)

        submit = st.form_submit_button(
            "Predict Diabetes Risk",
            use_container_width=True
        )

    if submit:
        patient = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose
        }])

        patient = pd.get_dummies(patient)
        patient = patient.reindex(columns=features, fill_value=0)

        try:
            patient_scaled = scaler.transform(patient)
            result = int(model.predict(patient_scaled)[0])

            st.subheader("Prediction Result")

            if result == 1:
                st.error("Model Prediction: Diabetes")
            else:
                st.success("Model Prediction: No Diabetes")

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(patient_scaled)[0][1]
                st.metric("Estimated Model Probability", f"{probability * 100:.2f}%")
                st.progress(float(probability))

            st.dataframe(patient, use_container_width=True)

            st.warning(
                "Educational use only. This model cannot diagnose diabetes. "
                "Consult a healthcare professional for medical advice."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# Dataset Explorer
elif page == "Dataset Explorer":

    st.subheader("📈 Dataset Explorer")

    st.write("Dataset shape:", df.shape)
    st.write("Column names:", df.columns.tolist())

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    column = st.selectbox("Select a column", numeric_columns)
    graph = st.selectbox("Graph type", ["Histogram", "Box Plot"])

    fig, ax = plt.subplots()

    if graph == "Histogram":
        ax.hist(df[column].dropna(), bins=25)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
    else:
        ax.boxplot(df[column].dropna(), vert=False)
        ax.set_xlabel(column)

    ax.set_title(f"{graph} of {column}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

# About Model
elif page == "About Model":

    st.subheader("🤖 About XGBoost")

    st.write("Algorithm: XGBoost Classifier")
    st.write("Number of input features:", len(features))

    st.markdown("""
    **Evaluation Metrics**
    - Accuracy: Overall correct predictions.
    - Precision: Correct positive predictions among all positive predictions.
    - Recall: Actual positive cases correctly identified.
    - F1 Score: Balance between precision and recall.
    """)

    st.subheader("Current Test Results")

    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Precision: {precision * 100:.2f}%")
    st.write(f"Recall: {recall * 100:.2f}%")
    st.write(f"F1 Score: {f1 * 100:.2f}%")

    st.write("Model input features:")
    st.write(features)
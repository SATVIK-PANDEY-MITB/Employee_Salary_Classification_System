import streamlit as st
import pandas as pd
import joblib

# Load trained model pipeline
model = joblib.load("xgboost_salary_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **more than 50K** or **less than or equal to 50K** based on various features.")

# Sidebar inputs
st.sidebar.header("👤 Enter Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Others"
])
education = st.sidebar.selectbox("Education", [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm",
    "Assoc-voc", "Doctorate", "Prof-school", "7th-8th", "12th", "10th"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced", "Separated",
    "Married-spouse-absent", "Widowed", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.radio("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "Others"
])
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
educational_num = st.sidebar.slider("Education Number (Years)", 1, 16, 10)
fnlwgt = st.sidebar.number_input("Fnlwgt", value=100000)

# Input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'educational-num': [educational_num],
    'fnlwgt': [fnlwgt]
})

st.subheader("📄 Input Data")
st.dataframe(input_df)

# Prediction
if st.button("🔍 Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    label = ">50K" if prediction == 1 else "≤50K"
    st.success(f"✅ Predicted Salary Class: **{label}**")

# Batch prediction
st.markdown("---")
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with the same 14 columns", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("🔎 Uploaded Data Sample:")
        st.dataframe(batch_data.head())

        batch_preds = model.predict(batch_data)
        batch_data["PredictedClass"] = [">50K" if p == 1 else "≤50K" for p in batch_preds]

        st.write("✅ Batch Predictions")
        st.dataframe(batch_data)

        # Download button
        csv = batch_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions CSV", csv, "salary_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

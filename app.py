# app.py

import streamlit as st
import pandas as pd
import joblib

# ✅ Import custom cleanup function
from utils import binary_cleanup

# ✅ Load trained pipeline after importing the function
model = joblib.load("attrition_model.joblib")

st.title("Employee Attrition Prediction")

with st.form("attrition_form"):
    age = st.slider("Age", 18, 60, 30)
    distance = st.slider("Distance From Home", 1, 30, 5)
    gender = st.selectbox("Gender", ["Male", "Female"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
        "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Age": age,
            "DistanceFromHome": distance,
            "Gender": gender,
            "OverTime": overtime,
            "BusinessTravel": business_travel,
            "Department": department,
            "EducationField": education_field,
            "JobRole": job_role,
            "MaritalStatus": marital_status,
            # Defaults for remaining fields
            "DailyRate": 800,
            "Education": 3,
            "EnvironmentSatisfaction": 3,
            "HourlyRate": 60,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobSatisfaction": 3,
            "MonthlyIncome": 5000,
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 2,
            "PercentSalaryHike": 15,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 8,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 4,
            "YearsInCurrentRole": 2,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2,
            "EmployeeCount": 1,
            "Over18": "Y",
            "StandardHours": 80,
            "EmployeeNumber": 1,
        }])

        prediction = model.predict(input_data)[0]
        label = "Yes (Will Leave)" if prediction == 1 else "No (Will Stay)"
        st.success(f"Prediction: {label}")
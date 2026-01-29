import streamlit as st
from helper import predict

st.set_page_config(page_title="Health Insurance Prediction", layout="wide")

st.title("🏥 Health Insurance Prediction App")

categorical_options = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Self-Employed'],
    'Region': ['Northwest', 'Southeast', 'Southwest'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure',
        'Diabetes & High blood pressure', 'Thyroid',
        'Heart disease', 'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold']
}

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', 18, 100, 30)
with row1[1]:
    number_of_dependants = st.number_input('Number of Dependants', 0, 10, 0)
with row1[2]:
    income_lakhs = st.number_input('Income (Lakhs)', 0, 200, 10)

with row2[0]:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])
with row2[1]:
    employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])
with row2[2]:
    gender = st.selectbox('Gender', categorical_options['Gender'])

with row3[0]:
    marital_status = st.selectbox('Marital Status', categorical_options['Marital Status'])
with row3[1]:
    bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])
with row3[2]:
    smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])

with row4[0]:
    region = st.selectbox('Region', categorical_options['Region'])
with row4[1]:
    medical_history = st.selectbox('Medical History', categorical_options['Medical History'])

input_dict = {
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

if st.button("🔮 Predict"):
    prediction = predict(input_dict)
    st.success(f"💰 Predicted Health Insurance Cost: {prediction}")

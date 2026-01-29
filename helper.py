import os
import pandas as pd
from joblib import load

# ---------- Load artifacts ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load(os.path.join(BASE_DIR, "artifacts", "xgboost_model.joblib"))

scaler_dict = load(os.path.join(BASE_DIR, "artifacts", "scaler.joblib"))
scaler = scaler_dict["scaler"]
cols_to_scale = scaler_dict["cols_to_scale"]

# ---------- Income level ----------
def get_income_level(income):
    if income < 10:
        return 1
    elif income < 25:
        return 2
    elif income < 40:
        return 3
    else:
        return 4


# ---------- Medical risk ----------
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0
    }

    diseases = medical_history.lower().split(" & ")
    total_score = sum(risk_scores.get(d, 0) for d in diseases)

    return total_score / 14


# ---------- Preprocessing ----------
def preprocess_input(input_dict):
    columns = [
        'age', 'number_of_dependants', 'income_level', 'income_lakhs',
        'insurance_plan', 'normalized_risk_score',
        'gender_Male',
        'region_Northwest', 'region_Southeast', 'region_Southwest',
        'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    df = pd.DataFrame(0, columns=columns, index=[0])

    # Numerical
    df['age'] = input_dict['Age']
    df['number_of_dependants'] = input_dict['Number of Dependants']
    df['income_lakhs'] = input_dict['Income in Lakhs']
    df['income_level'] = get_income_level(input_dict['Income in Lakhs'])

    # Insurance plan
    df['insurance_plan'] = {'Bronze': 1, 'Silver': 2, 'Gold': 3}[input_dict['Insurance Plan']]

    # Binary encodings
    if input_dict['Gender'] == 'Male':
        df['gender_Male'] = 1

    if input_dict['Marital Status'] == 'Unmarried':
        df['marital_status_Unmarried'] = 1

    if input_dict['Region'] == 'Northwest':
        df['region_Northwest'] = 1
    elif input_dict['Region'] == 'Southeast':
        df['region_Southeast'] = 1
    elif input_dict['Region'] == 'Southwest':
        df['region_Southwest'] = 1

    if input_dict['BMI Category'] == 'Obesity':
        df['bmi_category_Obesity'] = 1
    elif input_dict['BMI Category'] == 'Overweight':
        df['bmi_category_Overweight'] = 1
    elif input_dict['BMI Category'] == 'Underweight':
        df['bmi_category_Underweight'] = 1

    if input_dict['Smoking Status'] == 'Occasional':
        df['smoking_status_Occasional'] = 1
    elif input_dict['Smoking Status'] == 'Regular':
        df['smoking_status_Regular'] = 1

    if input_dict['Employment Status'] == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif input_dict['Employment Status'] == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    # Risk score
    df['normalized_risk_score'] = calculate_normalized_risk(
        input_dict['Medical History']
    )

    return df


# ---------- Prediction ----------
def predict(input_dict):
    df = preprocess_input(input_dict)

    # Scale trained columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # IMPORTANT: match model feature order
    df = df[model.feature_names_in_]

    prediction = model.predict(df)
    return int(prediction[0])

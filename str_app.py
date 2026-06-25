import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/best_model.pkl")

st.title("Medical Insurance Cost Prediction")

age = st.slider("Age",18,100,25)

sex = st.selectbox(
    "Gender",
    ["male","female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.slider(
    "Children",
    0,
    10,
    0
)

smoker = st.selectbox(
    "Smoker",
    ["yes","no"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

if st.button("Predict"):

    sex = 1 if sex=="male" else 0
    smoker = 1 if smoker=="yes" else 0

    region_dict = {
        "northeast":0,
        "northwest":1,
        "southeast":2,
        "southwest":3
    }

    region = region_dict[region]

    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    input_df = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region],
        "bmi_category":[bmi_category]
    })

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Insurance Cost : ${prediction:.2f}"
    )
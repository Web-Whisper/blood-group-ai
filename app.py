
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("AI Blood Group Predictor")

df = pd.read_csv("lab_dataset.csv")

X = df.drop("blood_group", axis=1)
y = df["blood_group"]

model = RandomForestClassifier()
model.fit(X, y)

st.subheader("Enter Lab Values")

hb = st.number_input("Hemoglobin")
rbc = st.number_input("RBC")
wbc = st.number_input("WBC")
platelets = st.number_input("Platelets")

if st.button("Predict"):
    pred = model.predict([[hb,rbc,wbc,platelets]])
    st.success(f"Predicted Blood Group: {pred[0]}")

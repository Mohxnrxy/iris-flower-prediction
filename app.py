import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Iris Flower Prediction", page_icon="🌸")

st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements below:")

sl = st.number_input("Sepal Length (cm)", value=5.1)
sw = st.number_input("Sepal Width (cm)", value=3.5)
pl = st.number_input("Petal Length (cm)", value=1.4)
pw = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict"):
    prediction = model.predict([[sl, sw, pl, pw]])

    classes = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Flower: **{classes[prediction[0]]}**")
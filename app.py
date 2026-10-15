import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Grocery Customer Satisfaction AI", page_icon="🛒", layout="wide")

MODEL_PATH = "models/customer_satisfaction_model.pkl"
DATA_PATH = "dataset/customer_satisfaction.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

st.title("🛒 Grocery Store Customer Satisfaction AI")
st.caption("Python + Machine Learning | Random Forest Classification")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Satisfied", int((df["Satisfaction"] == "Satisfied").sum()))
with col3:
    st.metric("Satisfaction Rate", f"{(df['Satisfaction'].eq('Satisfied').mean()*100):.1f}%")

st.divider()
st.subheader("🔮 Predict Customer Satisfaction")

left, right = st.columns(2)

with left:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    freq = st.selectbox("Shopping Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    amount = st.number_input("Amount Spent (₹)", min_value=0, max_value=100000, value=2000, step=100)
    quality = st.slider("Product Quality", 1, 5, 4)

with right:
    staff = st.slider("Staff Behavior", 1, 5, 4)
    clean = st.slider("Store Cleanliness", 1, 5, 4)
    waiting = st.number_input("Waiting Time (minutes)", min_value=0, max_value=180, value=10)
    billing = st.slider("Billing Experience", 1, 5, 4)
    complaints = st.number_input("Number of Complaints", min_value=0, max_value=20, value=0)

input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Shopping_Frequency": freq,
    "Amount_Spent": amount,
    "Product_Quality": quality,
    "Staff_Behavior": staff,
    "Store_Cleanliness": clean,
    "Waiting_Time": waiting,
    "Billing_Experience": billing,
    "Complaints": complaints
}])

if st.button("Predict Satisfaction", type="primary", use_container_width=True):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    prob = probabilities[list(classes).index(prediction)] * 100

    if prediction == "Satisfied":
        st.success(f"😊 Prediction: {prediction}")
    else:
        st.error(f"☹️ Prediction: {prediction}")

    st.metric("Model Confidence", f"{prob:.1f}%")

st.divider()
st.subheader("📊 Customer Data Analysis")

a, b = st.columns(2)
with a:
    st.write("### Satisfaction Distribution")
    st.bar_chart(df["Satisfaction"].value_counts())
with b:
    st.write("### Average Ratings")
    rating_cols = ["Product_Quality", "Staff_Behavior", "Store_Cleanliness", "Billing_Experience"]
    st.bar_chart(df[rating_cols].mean())

st.write("### Sample Dataset")
st.dataframe(df.head(20), use_container_width=True)

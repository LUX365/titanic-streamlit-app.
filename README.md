%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained logistic regression model
model = joblib.load('logistic_model.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival:")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 30)
SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare Paid", min_value=0.0, value=50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert text to numbers
Sex = 1 if Sex == "male" else 0
embark_map = {"C": 0, "Q": 1, "S": 2}
Embarked = embark_map[Embarked]

# Combine into a numpy array
features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])

if st.button("Predict"):
    prediction = model.predict(features)
    st.subheader("🎯 Result:")
    st.write("✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive")

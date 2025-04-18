# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict if they would have survived the Titanic disaster.")

# Sidebar for user input
st.sidebar.header("🔧 Passenger Input")

def user_input():
    pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    age = st.sidebar.slider("Age", 1, 80, 30)
    sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 5, 0)
    parch = st.sidebar.number_input("Parents/Children Aboard", 0, 5, 0)
    fare = st.sidebar.slider("Fare", 0.0, 500.0, 32.0)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "Q", "C"])

    sex = 0 if sex == "Male" else 1
    embarked_S = 1 if embarked == "S" else 0
    embarked_Q = 1 if embarked == "Q" else 0

    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S
    }

    return pd.DataFrame([data], columns=[
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'
    ])

input_df = user_input()

# Load and preprocess Titanic data for training
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
df.dropna(subset=['Embarked'], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Predict and display
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.success("🎉 The passenger **would have survived**.")
else:
    st.error("💀 The passenger **would not have survived**.")

st.subheader("Prediction Probability")
st.info(f"Survival: {prediction_proba[0][1]*100:.2f}%")

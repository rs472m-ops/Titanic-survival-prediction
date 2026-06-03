import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Titanic Survival Prediction")

# Read data
df = pd.read_csv("titanic.csv")

# Clean data
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Split data
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
trees = st.sidebar.slider("Number of Trees", 10, 300, 100)

model = RandomForestClassifier(
    n_estimators=trees,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.write("Accuracy:", round(accuracy * 100, 2), "%")

# User input
st.subheader("Passenger Details")

pclass = st.selectbox("Class", [1, 2, 3])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 100, 25)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)

# Prediction
if st.button("Predict"):

    sex = 0 if gender == "Male" else 1

    passenger = pd.DataFrame(
        [[pclass, sex, age, 0, 0, fare, 0]],
        columns=X.columns
    )

    prediction = model.predict(passenger)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
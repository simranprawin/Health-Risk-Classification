from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


app = Flask(__name__)


# Sample health-risk dataset
data = {
    "Age": [22, 25, 28, 31, 35, 38, 41, 45, 48, 52, 56, 60],
    "Glucose": [90, 95, 100, 105, 115, 125, 130, 140, 150, 160, 170, 180],
    "BloodPressure": [70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98],
    "BMI": [21, 22, 24, 25, 26, 27, 29, 31, 33, 35, 37, 39],
    "Risk": [
        "Healthy",
        "Healthy",
        "Healthy",
        "Healthy",
        "Healthy",
        "At Risk",
        "At Risk",
        "At Risk",
        "At Risk",
        "At Risk",
        "At Risk",
        "At Risk"
    ]
}

df = pd.DataFrame(data)

df["RiskLabel"] = df["Risk"].map({
    "Healthy": 0,
    "At Risk": 1
})

features = ["Age", "Glucose", "BloodPressure", "BMI"]

X = df[features]
y = df["RiskLabel"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    input_data = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            glucose = float(request.form["glucose"])
            blood_pressure = float(request.form["blood_pressure"])
            bmi = float(request.form["bmi"])

            input_data = {
                "age": age,
                "glucose": glucose,
                "blood_pressure": blood_pressure,
                "bmi": bmi
            }

            new_person = pd.DataFrame({
                "Age": [age],
                "Glucose": [glucose],
                "BloodPressure": [blood_pressure],
                "BMI": [bmi]
            })

            result = model.predict(new_person)[0]

            prediction = "At Risk" if result == 1 else "Healthy"

        except (ValueError, KeyError):
            error = "Please enter valid numbers in all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        input_data=input_data,
        accuracy=accuracy,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
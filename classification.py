import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# 1. Create the dataset
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


# 2. Convert the data into a DataFrame
df = pd.DataFrame(data)

print("DATASET")
print(df)


# 3. Convert text labels into numbers
risk_mapping = {
    "Healthy": 0,
    "At Risk": 1
}

df["RiskLabel"] = df["Risk"].map(risk_mapping)

print("\nLABEL MAPPING")
print(df[["Risk", "RiskLabel"]])


# 4. Select features and target
X = df[["Age", "Glucose", "BloodPressure", "BMI"]]
y = df["RiskLabel"]


# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 6. Create and train the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)


# 7. Make predictions
predictions = model.predict(X_test)

print("\nPREDICTIONS")
print(predictions)

print("\nACTUAL VALUES")
print(y_test.to_numpy())


# 8. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY")
print(f"{accuracy:.2%}")


# 9. Display the confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Healthy", "At Risk"],
    yticklabels=["Healthy", "At Risk"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()


# 10. Display the classification report
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Healthy", "At Risk"],
        zero_division=0
    )
)


# 11. Predict the risk for a new person
new_person = pd.DataFrame({
    "Age": [40],
    "Glucose": [135],
    "BloodPressure": [86],
    "BMI": [30]
})

new_prediction = model.predict(new_person)

if new_prediction[0] == 1:
    predicted_label = "At Risk"
else:
    predicted_label = "Healthy"

print("\nNEW PERSON PREDICTION")
print("Person details:")
print(new_person)
print("Predicted Risk:", predicted_label)
# 🩺 Health Risk Classification Dashboard

A machine learning web application that uses a **Decision Tree Classifier** to classify a sample health profile as **Healthy** or **At Risk** based on Age, Glucose, Blood Pressure, and BMI.

> ⚠️ **Educational Project:** This application is created for learning and demonstration purposes only. It is not a medical diagnosis or a substitute for professional medical advice.

---

## 📌 Project Overview

This project demonstrates a complete machine learning workflow:

- Creating a sample dataset
- Preparing and transforming data
- Converting categorical labels into numerical values
- Selecting input features and target values
- Splitting data into training and testing sets
- Training a Decision Tree classification model
- Making predictions
- Evaluating the model
- Building a Flask web application
- Creating an interactive prediction dashboard

The model uses the following input features:

- Age
- Glucose Level
- Blood Pressure
- BMI

The prediction classes are:

- **Healthy**
- **At Risk**

---

## ✨ Features

### 📊 Machine Learning

- Decision Tree Classification
- Train-test split
- Stratified data splitting
- Model accuracy calculation
- Confusion matrix
- Classification report
- Prediction for new input data

### 🌐 Web Application

- Flask-based backend
- Interactive health information form
- Health risk prediction
- Displays submitted information
- Displays model accuracy
- Error handling for invalid input
- Responsive dashboard interface

### 🎨 Dashboard

- Modern health-themed interface
- Prediction result card
- Healthy / At Risk visual indicators
- Model information
- Accuracy display
- Responsive design for smaller screens

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Decision Tree Classifier

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Web Development

- Flask
- HTML
- CSS

---

## 🧠 Machine Learning Model

The project uses a **Decision Tree Classifier**.

The model is configured with:

```python
DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
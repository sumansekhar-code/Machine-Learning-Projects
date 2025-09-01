Salary Prediction Project

This project is a Machine Learning web application built with Flask. It predicts the salary of an employee based on three factors:
Experience (years)
Test Score
Interview Score

The backend model is trained using Linear Regression on a small dataset (hiring.csv). The app provides two ways to get predictions:
Web Interface → Enter details in a form and get salary prediction instantly.
API Endpoint → Send a JSON request and receive the predicted salary.

Tech Stack:
Python (Flask, Pandas, NumPy, scikit-learn)
HTML (Jinja2) for frontend
Pickle for saving the trained model

How it Works:
Data is preprocessed (missing values handled, experience words converted to numbers).
A Linear Regression model is trained and saved as model.pkl.
Flask app (app.py) loads the model and serves predictions via web or API.

Quick Start:
# Train the model
python model.py  

# Run Flask app
python app.py  

App will be available at: http://127.0.0.1:5000/

Example API Usage:
import requests
url = "http://localhost:5000/predict_api"
payload = {"experience": 2, "test_score": 9, "interview_score": 6}
print(requests.post(url, json=payload).json())


Response:
{"prediction": 53713.45}

Future Scope:
Add advanced ML models for better accuracy
Deploy on Heroku / Render / AWS
Improve UI with Bootstrap or React
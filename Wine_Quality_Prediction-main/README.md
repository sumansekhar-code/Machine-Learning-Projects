# Wine Quality Prediction using Machine Learning

This project is a Flask-based web application that predicts the **quality of red wine** based on various physicochemical properties using a **K-Nearest Neighbors (KNN)** classification model.


## Problem Statement

Wine quality is influenced by several measurable factors such as acidity, sugar content, sulphates, etc. The goal of this project is to build a machine learning model that can predict wine quality on a scale of 3 to 8 based on these features.


## Machine Learning Workflow

- Data Source: `winequality-red.csv`  
- Data Cleaning and Log Transformation of Skewed Features  
- Class Balancing (Upsampling + Downsampling)  
- Feature Selection  
- Model: **K-Nearest Neighbors (KNN)** with `weights='distance'`  
- Evaluation on both training and test sets  
- Deployment using **Flask** web framework  


## Tech Stack

- **Python**, **NumPy**, **Pandas**, **Scikit-learn**
- **Matplotlib**, **Seaborn**, **Plotly**
- **Flask** for backend API
- **HTML/CSS/Bootstrap** for frontend
- **Pickle** for model serialization
- Deployed using **Gunicorn + Procfile** (ready for platforms like Heroku)



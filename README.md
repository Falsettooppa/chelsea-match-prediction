📘 Chelsea FC Match Outcome Prediction System
1. Project Overview

This project presents a machine learning–based system for predicting Chelsea Football Club match outcomes in the English Premier League. The system uses historical match data, rolling performance indicators, and betting odds to predict whether Chelsea will win, draw, or lose a given match.

The project was developed as a final-year undergraduate project and follows a complete data science workflow, including data preprocessing, feature engineering, model training, evaluation, and deployment through a web-based interface.

2. Objectives of the Project

The main objectives of this project are to:

i. develop a machine learning model capable of predicting Chelsea FC match outcomes;
ii. apply feature engineering techniques to capture recent team performance;
iii. evaluate the performance of different classification models;
iv. deploy the final prediction model as a usable system through a web interface.

3. Dataset Description

The dataset used in this project consists of English Premier League match data obtained from historical records. Chelsea FC matches were extracted across multiple seasons.

For each match, features were engineered using a rolling window of the previous five matches, ensuring that only pre-match information was used for prediction. This approach prevents data leakage and simulates real-world prediction scenarios.

4. Feature Engineering

The following categories of features were used:

i. match context features such as home or away status;
ii. rolling performance features including recent points, goals scored, goals conceded, goal difference, and win rate;
iii. betting odds representing pre-match market expectations.

These features were selected based on football domain knowledge and the methodology outlined in the project.

5. Machine Learning Models

Three models were implemented and evaluated:

i. Logistic Regression (baseline model);
ii. Random Forest classifier without betting odds;
iii. Random Forest classifier with betting odds (final model).

A time-aware train–test split was used to ensure realistic evaluation, with earlier matches used for training and later matches used for testing.

The final Random Forest model achieved the best performance and was selected for deployment.

6. System Implementation

The system was implemented using the Python programming language and the following tools:

Pandas and NumPy for data preprocessing

Scikit-learn for model training and evaluation

Joblib for saving and loading the trained model

Flask for building the web-based prediction interface

The system supports prediction through both a command-line interface and a web application.

7. Web Application

A Flask-based web application was developed to allow users to:

i. specify whether Chelsea is playing at home or away;
ii. enter betting odds for win, draw, and loss outcomes;
iii. provide the scorelines of Chelsea’s last five matches;
iv. obtain a predicted match outcome along with probability estimates and a confidence level.

This interface demonstrates the practical applicability of the developed prediction model.

8. Project Structure

The project directory is organised as follows:

chelsea-match-prediction/
│
├── web/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── src/
│   ├── data preparation and model training scripts
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   └── final_rf_model.joblib
│
├── requirements.txt
├── Procfile
└── README.md
9. How to Run the Project Locally
Step 1: Install dependencies
pip install -r requirements.txt
Step 2: Run the web application
python web/app.py
Step 3: Access the application

Open a browser and visit:

http://127.0.0.1:5000
10. Deployment

The web application was deployed using Render, allowing public access to the prediction system. Deployment was carried out using a GitHub repository, a requirements.txt file, and a Procfile to define the application start command.

11. Limitations and Future Work

Although the system achieved encouraging results, several limitations remain. These include the restriction of the dataset to a single club, the absence of opponent-specific features, and the inherent unpredictability of football matches.

Future improvements may include incorporating opponent form, expected goals (xG) statistics, and expanding the system to support multiple teams.
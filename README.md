# mentorness_ml_1
Internship track
# Salary Predictions of Data Professions
Welcome to the Salary Predictions of Data Professions project! This project aims to predict the salaries of data professionals based on various factors such as experience, job role, and performance ratings. The project includes data analysis, feature engineering, machine learning model development, and deployment of the predictive model as a web application.
# Project Structure
    • app.py: This file contains the Flask web application code for serving the predictive model and handling HTTP requests.
    • salary_prediction_pipeline_lr.pkl: This is the serialized machine learning model (e.g., trained with scikit-learn) used for predicting salaries.
    • static/: This directory contains static files such as CSS stylesheets, JavaScript files,  used for styling the web application.
    • templates/: This directory contains HTML templates used for rendering the web pages of the application.
# Getting Started
To run the Flask web application locally, follow these steps(make sure to have flask installed):
    1. Install Python 3.x on your machine if you haven't already.
    2. Clone this repository to your local machine.
    3. Navigate to the project directory.
    4. Create a virtual environment: python3 -m venv venv
    5. Activate the virtual environment:
        ◦ On Windows: venv\Scripts\activate
        ◦ On macOS and Linux: source venv/bin/activate
    6. Run the Flask application: python app.py
    7. Open your web browser and navigate to http://localhost:5000 to access the application.
# Usage
    • Once the Flask application is running, you can use the web interface to input data for predicting salaries of data professionals.
    • Enter the required information such as job role, experience, age, and ratings, then click the "Predict" button to see the predicted salary.
    • The application will display the predicted salary based on the input provided.


# mentorness_ml_2
Internship track
# Fast Tag Fraud Detection
Welcome to the Fastag Fraud Detection System project! This project tackles the challenge of identifying fraudulent activities within the Fastag toll payment system in India.

Our goal is to develop a robust system that can analyze real-world Fastag transaction data and accurately classify transactions as either legitimate or fraudulent.



# File Structure

    *requirements.txt        # Text file containing required Python libraries
    *mentorness_ml_2_final.ipynb  # Jupyter notebook containing EDA, feature engineering, model development, and evaluation
    *app.py                   # Python script for the Streamlit web application for real-time fraud prediction
    *svm_fast_best.pkl       # Pickled model file containing the best performing SVM model
    *svm_fast.pkl            # Pickled model file containing a trained SVM model
    *FastagFraudDetection.csv  # CSV file containing the fastag transaction data
# Project Overview
    *Data Exploration and Preprocessing (mentorness_ml_2_final.ipynb):

Loads the Fastag fraud detection dataset (FastagFraudDetection.csv).
Performs exploratory data analysis (EDA) to understand the data distribution, identify missing values, and analyze relationships between features.
Preprocesses the data by handling missing values, encoding categorical features, and feature engineering (e.g., extracting time-based features).
Splits the data into training and testing sets.

    *Model Development and Evaluation (mentorness_ml_2_final.ipynb):

Implements and trains various machine learning models like Decision Tree, Random Forest, AdaBoost, XGBoost, KNN, and SVM for fraud classification.
Performs hyperparameter tuning using GridSearchCV to find the best performing model configuration for each algorithm.
Evaluates the models using metrics like accuracy, precision, recall, F1-score, and ROC AUC curve.
Selects the best performing model based on evaluation metrics.

    *Model Deployment (app.py):

Creates a Streamlit web application (app.py) for real-time fraud prediction.
Loads the best performing model saved as a pickle file (svm_fast_best.pkl).
Defines the features used for model prediction.
Provides a user interface where users can input transaction details.
Uses the loaded model to predict whether a transaction is fraudulent based on the user input.
Displays the predicted fraud probability or classification.

# Requirements
The project requires the following Python libraries:

    pandas
    scikit-learn
    streamlit
    pickle
You can install these libraries using the pip command:

# Bash

    *pip install pandas scikit-learn streamlit pickle
    

# How to Use
Clone this repository to your local machine.
Install the required libraries using the command mentioned above.
1.Run the Jupyter notebook (mentorness_ml_2_final.ipynb) to explore the data, train the models, and evaluate their performance.
2.Run the Streamlit application (app.py) using the command:
# Bash
    streamlit run app.py

This will launch the Streamlit app in your web browser, allowing you to interact with the fraud prediction model.

# Note
The Jupyter notebook (mentorness_ml_2_final.ipynb) assumes the project files are located in the same directory as the notebook. The file paths for data loading and model saving might need adjustments depending on your specific setup.

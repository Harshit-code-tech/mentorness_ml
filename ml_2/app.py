from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load('/home/hgidea/Desktop/Coding/Python/internship/mentorness/ml_2/fastag_fraud_detection_pipeline.pkl')

# Initialize Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    # Convert JSON data to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    # Assuming 'input_df' is your input data for prediction
    print(input_df.shape)  # This should output (num_samples, 23)

    # Predict using the pipeline
    prediction = pipeline.predict(input_df)
    # Return the prediction
    return jsonify({'fraud': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

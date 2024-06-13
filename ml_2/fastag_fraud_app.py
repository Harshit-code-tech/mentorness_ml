import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('/home/hgidea/Desktop/Coding/Python/internship/mentorness/ml_2/fastag_fraud_detection_pipeline.pkl')

st.title("Fastag Fraud Detection")

st.write("""
## Predict whether a Fastag transaction is fraudulent.
""")


def user_input_features():
    Transaction_Amount = st.number_input("Transaction Amount", min_value=0.0)
    Amount_paid = st.number_input("Amount Paid", min_value=0.0)
    Vehicle_Speed = st.number_input("Vehicle Speed", min_value=0.0)
    Hour = st.number_input("Hour", min_value=0, max_value=23)
    DayOfWeek = st.number_input("Day of the Week", min_value=0, max_value=6)
    Month = st.number_input("Month", min_value=1, max_value=12)
    Transaction_Discrepancy = Transaction_Amount - Amount_paid

    Vehicle_Type_Car = st.selectbox("Vehicle Type - Car", [0, 1])
    Vehicle_Type_Bus = st.selectbox("Vehicle Type - Bus", [0, 1])
    Vehicle_Type_Truck = st.selectbox("Vehicle Type - Truck", [0, 1])
    Vehicle_Type_Van = st.selectbox("Vehicle Type - Van", [0, 1])
    Lane_Type_Fastag = st.selectbox("Lane Type - Fastag", [0, 1])
    Lane_Type_Cash = st.selectbox("Lane Type - Cash", [0, 1])
    TollBoothID_1 = st.selectbox("TollBooth ID - 1", [0, 1])
    TollBoothID_2 = st.selectbox("TollBooth ID - 2", [0, 1])
    TollBoothID_3 = st.selectbox("TollBooth ID - 3", [0, 1])
    TollBoothID_4 = st.selectbox("TollBooth ID - 4", [0, 1])

    data = {
        'Transaction_Amount': Transaction_Amount,
        'Amount_paid': Amount_paid,
        'Vehicle_Speed': Vehicle_Speed,
        'Hour': Hour,
        'DayOfWeek': DayOfWeek,
        'Month': Month,
        'Transaction_Discrepancy': Transaction_Discrepancy,
        'Vehicle_Type_Car': Vehicle_Type_Car,
        'Vehicle_Type_Bus': Vehicle_Type_Bus,
        'Vehicle_Type_Truck': Vehicle_Type_Truck,
        'Vehicle_Type_Van': Vehicle_Type_Van,
        'Lane_Type_Fastag': Lane_Type_Fastag,
        'Lane_Type_Cash': Lane_Type_Cash,
        'TollBoothID_1': TollBoothID_1,
        'TollBoothID_2': TollBoothID_2,
        'TollBoothID_3': TollBoothID_3,
        'TollBoothID_4': TollBoothID_4
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

if st.button('Predict'):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    st.write('Fraudulent' if prediction[0] else 'Not Fraudulent')

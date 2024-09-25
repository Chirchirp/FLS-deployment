from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    age = data['age']
    contract_type = data['Contract Type']
    monthly_charge = data['Monthly Charge']
    tenure = data['Tenure in Months']
    internet_service = data['Internet Service']

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'Contract Type': [contract_type],
        'Monthly Charge': [monthly_charge],
        'Tenure in Months': [tenure],
        'Internet Service': [internet_service]
    })

    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert prediction to a more readable format
    churn_prediction = 'Yes' if prediction[0] == 1 else 'No'

    return jsonify({'Churn Prediction': churn_prediction})

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*", "methods": "*", "headers": "*"}}

# Load the CSV file
data = pd.read_csv('Updated_Kwara_State_Crops_Calendar.csv')

# Encode categorical variables
label_encoder_location = LabelEncoder()
label_encoder_crop = LabelEncoder()

data['location_encoded'] = label_encoder_location.fit_transform(data['Location'])
data['crop_encoded'] = label_encoder_crop.fit_transform(data['Crop'])
data['planting_month'] = data['Planting Time']

# Define features and target variables 
X_planting = data[['location_encoded', 'crop_encoded']]
y_planting = data['planting_month']

X_development = data[['location_encoded', 'crop_encoded']]
y_development = data['Development Days']

# Split data for planting month prediction
X_train_planting, X_test_planting, y_train_planting, y_test_planting = train_test_split(X_planting, y_planting, test_size=0.2, random_state=42)

# Split data for development days prediction
X_train_development, X_test_development, y_train_development, y_test_development = train_test_split(X_development, y_development, test_size=0.2, random_state=42)

# Initialize and train the model for planting month
model_planting_month = DecisionTreeClassifier(random_state=42)
model_planting_month.fit(X_train_planting, y_train_planting)

# Initialize and train the development days model
model_development_days = DecisionTreeClassifier(random_state=42)
model_development_days.fit(X_train_development, y_train_development)

# Predict on the test set for planting month
y_pred_planting = model_planting_month.predict(X_test_planting)

# Calculate accuracy for planting month
accuracy_planting = accuracy_score(y_test_planting, y_pred_planting)
print(f'Planting month model accuracy: {accuracy_planting * 100:.2f}%')

# Predict on the test set for development days
y_pred_development = model_development_days.predict(X_test_development)

# Calculate accuracy for development days
accuracy_development = accuracy_score(y_test_development, y_pred_development)
print(f'Development days model accuracy: {accuracy_development * 100:.2f}%')

# Create prediction function for planting month
def predict_planting_month(location, crop):
    # Encode the input
    location_encoded = label_encoder_location.transform([location])[0]
    crop_encoded = label_encoder_crop.transform([crop])[0]

    # Create input array for prediction
    input_data = [[location_encoded, crop_encoded]]

    # Make prediction for planting month
    predicted_month = model_planting_month.predict(input_data)[0]
    
    return predicted_month

def predict_harvesting_date(location, crop, planting_date):
    # Encode the input
    location_encoded = label_encoder_location.transform([location])[0]
    crop_encoded = label_encoder_crop.transform([crop])[0]
    planting_date = pd.to_datetime(planting_date)
    
    # Create input array for prediction
    input_data = [[location_encoded, crop_encoded]]

    # Predict development days
    predicted_development_days = model_development_days.predict(input_data)[0]

    # Calculate harvesting date
    harvesting_date = planting_date + pd.to_timedelta(predicted_development_days, unit='D')
    
    return harvesting_date.strftime('%Y-%m-%d')

# Define routes for the API
@app.route('/predict_planting_month', methods=['POST'])
def api_predict_planting_month():
    data = request.get_json()
    location = data['location']
    crop = data['crop']
    predicted_month = predict_planting_month(location, crop)
    return jsonify({'planting_month': predicted_month})

@app.route('/predict_harvesting_date', methods=['POST'])
def api_predict_harvesting_date():
    data = request.get_json()
    location = data['location']
    crop = data['crop']
    planting_date = data['planting_date']
    predicted_harvesting_date = predict_harvesting_date(location, crop, planting_date)
    return jsonify({'harvesting_date': predicted_harvesting_date})

if __name__ == '__main__':
    app.run(debug=True)

# Example usage of the prediction functions
location_input = 'Kwara'
crop_input = 'Rice'
planting_date_input = '2024-05-01'

# Predict planting month
predicted_planting_month = predict_planting_month(location_input, crop_input)

# Predict harvesting date
predicted_harvesting_date = predict_harvesting_date(location_input, crop_input, planting_date_input)

print(f'The predicted planting month for {crop_input} in {location_input} is: {predicted_planting_month}')
print(f'The predicted harvesting date for {crop_input} in {location_input} is: {predicted_harvesting_date}')

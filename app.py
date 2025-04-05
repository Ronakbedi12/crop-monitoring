from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Dummy model for crop yield prediction (just a placeholder)
# In a real-world scenario, train a proper model based on historical data
model = LinearRegression()

# Simulated training data
X = np.array([[200, 20, 6.5, 60],
              [300, 25, 7.0, 55],
              [250, 22, 6.8, 70],
              [400, 28, 7.5, 65]])
y = np.array([2.5, 3.0, 2.8, 3.5])  # Crop yield in tons per acre

# Train the model
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract data from the request
    rainfall = float(data['rainfall'])
    temperature = float(data['temperature'])
    soil_ph = float(data['soil_ph'])
    humidity = float(data['humidity'])

    # Prepare input for prediction
    input_features = np.array([[rainfall, temperature, soil_ph, humidity]])

    # Make prediction using the trained model
    prediction = model.predict(input_features)

    # Return the predicted value as JSON
    return jsonify({'yield': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
 
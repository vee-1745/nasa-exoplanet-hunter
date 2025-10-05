from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the upgraded, pre-trained machine learning model
model = joblib.load('model/exoplanet_model.joblib')

# Define the route for the main home page
@app.route('/')
def home():
    # Render the index.html template
    return render_template('index.html')

# Define the route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user inputs from the HTML form
    # We use float() to convert the string inputs to numbers
    features = [float(x) for x in request.form.values()]
    
    # Create a pandas DataFrame from the features, with the correct column names
    feature_names = ['period', 'duration', 'depth', 'planet_radius', 
                     'stellar_temp', 'stellar_gravity', 'stellar_radius']
    input_data = pd.DataFrame([features], columns=feature_names)
    
    # Use the model to make a prediction and get probabilities
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Determine the result text and confidence score
    if prediction == 1:
        result_text = "CONFIRMED PLANET 🪐"
        confidence = f"{prediction_proba[1]*100:.2f}%"
    else:
        result_text = "FALSE POSITIVE ❌"
        confidence = f"{prediction_proba[0]*100:.2f}%"
        
    # Render the index.html page again, but this time with the prediction results
    return render_template('index.html', prediction_text=result_text, confidence_text=f"Confidence: {confidence}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
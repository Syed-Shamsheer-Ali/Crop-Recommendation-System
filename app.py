from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scalers
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('minmaxscaler.pkl', 'rb') as f:
    minmax = pickle.load(f)
with open('standscaler.pkl', 'rb') as f:
    standard = pickle.load(f)


crop_labels = {
    0: 'rice',
    1: 'maize',
    2: 'chickpea',
    3: 'kidneybeans',
    4: 'pigeonpeas',
    5: 'mothbeans',
    6: 'mungbean',
    7: 'blackgram',
    8: 'lentil',
    9: 'pomegranate',
    10: 'banana',
    11: 'mango',
    12: 'grapes',
    13: 'watermelon',
    14: 'muskmelon',
    15: 'apple',
    16: 'orange',
    17: 'papaya',
    18: 'coconut',
    19: 'cotton',
    20: 'jute',
    21: 'coffee'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form and convert to float
        features = []
        for feature_name in ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']:
            val = request.form.get(feature_name, '').strip()
            if val == '':
                return render_template('index.html', prediction_text="Error: All fields are required.")
            features.append(float(val))

        # Scale features
        features_np = np.array([features])
        scaled = standard.transform(minmax.transform(features_np))

        # Predict
        prediction_int = model.predict(scaled)[0]

        # Map prediction to crop name
        crop_name = crop_labels.get(prediction_int, "Unknown crop")

        # Capitalize crop name and display
        return render_template('index.html', prediction_text=f"Recommended Crop: {crop_name.capitalize()}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
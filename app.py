from flask import Flask, request, jsonify
import pickle
import numpy as np
from feature import FeatureExtraction  # Ensure this import is correct

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get("url", "")

        # Extract numerical features
        features = FeatureExtraction(url)
        
        if not isinstance(features, list):  # Ensure it's a list
            features = list(features)
        
        # Convert to NumPy array and reshape for the model
        features = np.array(features).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(features)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

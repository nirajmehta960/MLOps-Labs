from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, static_folder='statics')

# California housing feature names (order must match training)
FEATURE_NAMES = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

# Load the trained model
model = joblib.load('housing_model.pkl')  # Replace with actual model file if different


@app.route('/')
def home():
    return "Welcome to the California Housing API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data or JSON data
            data = request.form if request.form else request.get_json()

            # Extract feature values from request
            values = [float(data.get(f, 0)) for f in FEATURE_NAMES]
            X = np.array(values).reshape(1, -1)

            # Perform the prediction
            pred = model.predict(X)[0]

            # Target is median house value in hundreds of thousands -> convert to full USD
            value_usd = int(round(float(pred) * 100_000))
            formatted = f"${value_usd:,}"

            # Return the predicted value in the response
            # Use jsonify() instead of json.dumps() in Flask
            return jsonify({
                "predicted_value_usd": value_usd,
                "formatted_usd": formatted
            })
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html', features=FEATURE_NAMES)
    else:
        return "Unsupported HTTP method"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)

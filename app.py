from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][int(prediction)]
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(round(prob, 3)),
            'result': 'Diabetic' if prediction else 'Not Diabetic'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

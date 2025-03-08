from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("student_exam_rf_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [data["study_hours"], data["sleep_hours"], data["attendance_rate"], data["previous_scores"]]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"predicted_exam_score": int(round(prediction))})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
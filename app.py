from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Label mapping (adjust if your model uses 0/1/2 or strings)
LABEL_MAP = {
    0: 'High',
    1: 'Low',
    2: 'Medium',
    'Low': 'Low',
    'Medium': 'Medium',
    'High': 'High'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        sleep_hours     = float(data['sleep_hours'])
        stress_level    = float(data['stress_level'])
        workload_rating = float(data['workload_rating'])
        attendance      = float(data['attendance'])

        features = np.array([[sleep_hours, stress_level, workload_rating, attendance]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0].tolist()

        label = LABEL_MAP.get(prediction, str(prediction))

        return jsonify({
            'success': True,
            'burnout_risk': label,
            'probabilities': {
                'High': round(proba[0] * 100, 1),
                'Low': round(proba[1] * 100, 1),
                'Medium': round(proba[2] * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

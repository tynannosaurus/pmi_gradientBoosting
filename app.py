from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

# Load the model
model = joblib.load('gradient_boosting_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

    })

if __name__ == '__main__':
    app.run(debug=True)

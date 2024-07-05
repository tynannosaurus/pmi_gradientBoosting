from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')

# Dummy variables for bootstrapping (You need to replace these with actual training data)
X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    
    # Standardise the features using the loaded scaler
    df_scaled = scaler.transform(df)

    # Make the main prediction
    prediction = model.predict(df_scaled)

    # Bootstrapping to estimate prediction intervals
    n_bootstraps = 100
    predictions_bootstrap = np.zeros((n_bootstraps, len(df_scaled)))

    for i in range(n_bootstraps):
        # Sample with replacement from the training data
        X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, random_state=i)
        model_bootstrap = GradientBoostingRegressor(n_estimators=100, max_depth=7, min_samples_leaf=2, min_samples_split=2, random_state=42)
        model_bootstrap.fit(X_train_bootstrap, y_train_bootstrap)
        predictions_bootstrap[i] = model_bootstrap.predict(df_scaled)

    # Calculate the 95% prediction interval
    lower_bound = np.percentile(predictions_bootstrap, 2.5, axis=0)
    upper_bound = np.percentile(predictions_bootstrap, 97.5, axis=0)

    return jsonify({
        'prediction': prediction.tolist(),
        'prediction_interval': {
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist()
        }
    })

if __name__ == '__main__':
    app.run(debug=True)

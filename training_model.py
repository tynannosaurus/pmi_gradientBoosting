import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load your data
data = pd.read_csv('/Users/tynanp/Documents/pmi_gradientBoosting/data.csv')

# Rename the 'Ambient ' column to 'Ambient' if necessary
data.rename(columns={'Ambient ': 'Ambient'}, inplace=True)

# Prepare the feature matrix and target vector
X = data[['Max', 'Min', 'Average', 'Ambient', 'Humidity']]
y = data['PMI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the gradient boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=7, min_samples_leaf=2, min_samples_split=2, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate the gradient boosting model
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Mean Squared Error: {mse_gb}")
print(f"R-squared: {r2_gb}")

# Save the trained model to a file
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
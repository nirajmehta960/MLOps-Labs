# Import necessary libraries
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

if __name__ == '__main__':
    # Load California housing (8 features, regression)
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R²:   {r2:.4f}")

    # Save the model (use output/ when mounted so it appears on host)
    os.makedirs('output', exist_ok=True)
    model_path = os.path.join('output', 'housing_model.pkl')
    joblib.dump(model, model_path)

    print(f"Model training was successful. Saved as {model_path}")

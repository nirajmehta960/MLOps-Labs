import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

if __name__ == '__main__':
    # Load the California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print(f"Test RMSE: {(mean_squared_error(y_test, y_pred)) ** 0.5:.4f}")
    print(f"Test MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Test R²:   {r2_score(y_test, y_pred):.4f}")

    # Save the trained model
    joblib.dump(model, 'housing_model.pkl')

    # Also save to output folder if it exists (for volume mounts)
    if os.path.isdir('output'):
        joblib.dump(model, 'output/housing_model.pkl')
        print("Model also saved to output/housing_model.pkl")
    else:
        print("Model was trained and saved as housing_model.pkl")

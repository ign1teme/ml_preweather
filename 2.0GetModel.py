from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def GetModel(X_train, y_train, X_test, y_test):
    # Define the RandomForestRegressor for multi-output regression
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    predictions = model.predict(X_test)

    # Evaluate the model performance using Mean Absolute Error
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")

    # Save the model to a file
    joblib.dump(model, 'weather_prediction_model.pkl')

    return predictions

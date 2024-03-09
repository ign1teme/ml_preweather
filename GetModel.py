from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from ProcessData import ProcessData
import joblib

def GetModel():
    # define data
    X_train, X_test, y_train, y_test = ProcessData()

    # define model
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # fitting the model
    model.fit(X_train, y_train)

    # predicting the model
    predictions = model.predict(X_test)

    # evaluating the model
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")

    # save the model
    joblib.dump(model, 'weather_prediction_model.pkl')

    return predictions

from GetModel import GetModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error
from ProcessData import ProcessData

X_train, X_test, y_train, y_test = ProcessData()

# define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# define meta learner
meta_learner = LinearRegression()

# define stacking model
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

# fitting the model
stacking_regressor.fit(X_train, y_train)
predictions = GetModel()
# prdicting the model
predictions = stacking_regressor.predict(X_test)

# evaluating the model
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error with Stacking: {mae}")


# Predictions for April 2024
print("Predictions for April 2024:")
print(predictions)

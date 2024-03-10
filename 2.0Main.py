from ProcessData import ProcessData
from GetModel import GetModel

# Prepare the data
X_train, X_test, y_train, y_test = ProcessData()

# Train the model and get predictions
predictions = GetModel(X_train, y_train, X_test, y_test)

# Print the predictions
for idx, (temp, wind_dir, wind_speed) in enumerate(predictions):
    print(f"Prediction {idx + 1}: Temperature = {temp:.2f}°C, Wind Direction = {wind_dir:.2f}°, Maximum Wind Speed = {wind_speed:.2f} m/s")

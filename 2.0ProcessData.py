import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

def ProcessData():
    # Load the data
    train_df = pd.read_excel('train.xlsx')  # Assuming Excel files are in the same directory as the script
    test_df = pd.read_excel('test.xlsx')

    # Replace non-numeric entries with NaN for imputation
    numeric_cols = ['Average temperature [°C]', 'Average wind direction [°]', 'Maximum wind speed [m/s]']
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # Impute missing values with mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

    # Feature selection
    feature_cols = ['Year', 'Month', 'Day']  # Update this if there are other numeric feature columns
    X_train = train_df[feature_cols]
    y_train = train_df[numeric_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[numeric_cols]

    return X_train, X_test, y_train, y_test

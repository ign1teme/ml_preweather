import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

def ProcessData():
    # Load the data
    train_df = pd.read_excel('train.xlsx')
    test_df = pd.read_excel('test.xlsx')

    # Convert date and time columns to datetime
    train_df['DateTime'] = pd.to_datetime(train_df['Year'].astype(str) + '/' + train_df['Month'].astype(str) + '/' + train_df['Day'].astype(str) + ' ' + train_df['Time [Local time]'])
    test_df['DateTime'] = pd.to_datetime(test_df['Year'].astype(str) + '/' + test_df['Month'].astype(str) + '/' + test_df['Day'].astype(str) + ' ' + test_df['Time [Local time]'])

    # Extract time features
    train_df['Year'] = train_df['DateTime'].dt.year
    train_df['Month'] = train_df['DateTime'].dt.month
    train_df['Day'] = train_df['DateTime'].dt.day
    train_df['Hour'] = train_df['DateTime'].dt.hour

    test_df['Year'] = test_df['DateTime'].dt.year
    test_df['Month'] = test_df['DateTime'].dt.month
    test_df['Day'] = test_df['DateTime'].dt.day
    test_df['Hour'] = test_df['DateTime'].dt.hour

    # Replace non-numeric entries with NaN for imputation
    numeric_cols = ['Average temperature [°C]', 'Average wind direction [°]', 'Maximum wind speed [m/s]']
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # Impute missing values using IterativeImputer
    imputer = IterativeImputer(random_state=42)
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

    # Feature selection
    feature_cols = ['Year', 'Month', 'Day', 'Hour']
    X_train = train_df[feature_cols]
    y_train = train_df[numeric_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[numeric_cols]

    return X_train, X_test, y_train, y_test
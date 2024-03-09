import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def ProcessData():
    # read data
    train_df = pd.read_excel("train.xlsx")
    test_df = pd.read_excel("test.xlsx")

    # emerage data
    full_df = pd.concat([train_df, test_df])

    # time feature
    cols_to_convert = ['Average temperature [°C]', 'Average wind direction [°]', 'Maximum wind speed [m/s]']
    for col in cols_to_convert:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # missing value
    imputer = SimpleImputer(strategy="mean")
    full_df[cols_to_convert] = imputer.fit_transform(full_df[cols_to_convert])

    # drop non-numeric features
    non_numeric_features = ['Observation station', 'Time [Local time]']
    full_df.drop(non_numeric_features, axis=1, inplace=True)

    # split data
    train = full_df[full_df['Year'] < 2024]
    test = full_df[full_df['Year'] >= 2024]

    # split features and target
    X_train = train.drop('Average temperature [°C]', axis=1)
    y_train = train['Average temperature [°C]']
    X_test = test.drop('Average temperature [°C]', axis=1)
    y_test = test['Average temperature [°C]']

    return X_train, X_test, y_train, y_test

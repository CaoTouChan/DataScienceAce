from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_cleaning_strategies import *


def load_data(file_path):
    """
    Load data from a specified CSV file path.

    :param file_path: str, path to the data file
    :return: pandas DataFrame
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def clean_data(data, config=None, custom_function=None):
    """
    # TODO: Maybe change to multiple strategies instead of one ?
    A unified function to clean data using various strategies. It can accept predefined strategies using
    a config dictionary or a custom cleaning function.

    :param data: pandas DataFrame - The input data that needs to be cleaned.
    :param config: dict, optional - A configuration dictionary that specifies which cleaning strategy to use and its parameters.
                   Example:
                   {
                       'strategy': 'missing_values',
                       'method': 'fill',
                       'value': 0
                   }
    :param custom_function: function, optional - A custom function that takes a DataFrame and returns a cleaned DataFrame.

    :return: pandas DataFrame - The cleaned data.

    Usage:

    1. Using predefined strategies with config:
    clean_data(data, config={'strategy': 'missing_values', 'method': 'fill', 'value': 0})

    2. Using a custom cleaning function:
    def my_cleaning_function(df):
        return df.drop_duplicates()
    clean_data(data, custom_function=my_cleaning_function)
    """
    cleaner = None
    if custom_function:
        cleaner = DataCleaner(CustomFunctionStrategy(custom_function))
    elif config:
        if config['strategy'] == 'missing_values':
            cleaner = DataCleaner(MissingValuesStrategy(method=config.get('method', 'drop'), value=config.get('value')))
        # ... more strategies based on config
        elif config['strategy'] == 'outliers':
            cleaner = DataCleaner(OutliersStrategy(method=config.get('method', 'remove'),
                                                   threshold=config.get('threshold', 1.5),
                                                   replace_with=config.get('replace_with')))

    if cleaner:
        cleaned_data = cleaner.clean_data(data)
    else:
        # TODO Default cleaning strategy
        cleaned_data = data.dropna()

    return cleaned_data


def preprocess_data(data, target_column, test_size=0.2, random_state=42):
    """
    Preprocess the cleaned data for model training.

    :param data: cleaned pandas DataFrame
    :param target_column: str, name of the target variable column
    :param test_size: float, proportion of the dataset to include in the test split
    :param random_state: int, random_state is the seed used by the random number generator
    :return: X_train, X_test, y_train, y_test (training and testing data)
    """
    # Separating features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardizing the features as an example
    # TODO: Adjust preprocessing steps based on your data and requirements
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data preprocessed successfully.")
    return X_train, X_test, y_train, y_test

import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_data, clean_data, preprocess_data


def test_load_data():
    # Assuming load_data returns a pandas DataFrame
    file_path = "data/raw/test/data.csv"
    data = load_data(file_path)
    assert not data.empty, "Data should not be empty"


def test_clean_data_missing_values_drop():
    data = pd.DataFrame({
        'column_1': [1, 2, np.nan, 4],
        'column_2': [5, 6, 7, 8]
    })
    config = {'strategy': 'missing_values', 'method': 'drop'}
    cleaned_data = clean_data(data, config=config)
    assert cleaned_data.isnull().sum().sum() == 0
    assert len(cleaned_data) == 3


def test_clean_data_missing_values_fill():
    data = pd.DataFrame({
        'column_1': [1, 2, np.nan, 4],
        'column_2': [5, 6, 7, 8]
    })
    config = {'strategy': 'missing_values', 'method': 'fill', 'value': 0}
    cleaned_data = clean_data(data, config=config)
    assert cleaned_data.isnull().sum().sum() == 0
    assert cleaned_data.at[2, 'column_1'] == 0


def test_clean_data_outliers_remove():
    data = pd.DataFrame({
        'column_1': [1, 2, 100, 4],
        'column_2': [5, 6, 7, 8]
    })
    config = {'strategy': 'outliers', 'method': 'remove', 'threshold': 1.5}
    cleaned_data = clean_data(data, config=config)
    assert 100 not in cleaned_data['column_1'].values


def test_clean_data_outliers_replace():
    data = pd.DataFrame({
        'column_1': [1, 2, 100, 4],
        'column_2': [5, 6, 7, 8]
    })
    config = {'strategy': 'outliers', 'method': 'replace', 'threshold': 1.5, 'replace_with': 99}
    cleaned_data = clean_data(data, config=config)
    assert cleaned_data.at[2, 'column_1'] == 99


def test_clean_data_custom_function():
    data = pd.DataFrame({
        'column_1': [1, 2, np.nan, 4],
        'column_2': [5, 6, 7, 8]
    })

    def custom_function(df):
        df = df.fillna(0)
        return df

    cleaned_data = clean_data(data, custom_function=custom_function)
    assert cleaned_data.isnull().sum().sum() == 0
    assert cleaned_data.at[2, 'column_1'] == 0


def test_preprocess_data():
    # Assuming preprocess_data standardizes the data
    # TODO: Implement tests for transform_features
    pass

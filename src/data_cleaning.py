# src/data_cleaning.py

from abc import ABC, abstractmethod
import pandas as pd
from scipy.stats import zscore
import numpy as np


class DataCleaningStrategy(ABC):
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame.

        :param data: pandas DataFrame
        :return: cleaned pandas DataFrame
        """
        pass


class MissingValuesStrategy(DataCleaningStrategy):
    def __init__(self, method='drop', value=None):
        self.method = method
        self.value = value

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.method == 'drop':
            cleaned_data = data.dropna()
        elif self.method == 'fill':
            cleaned_data = data.fillna(self.value)
        # ... more logic based on self.method
        return cleaned_data


class OutliersStrategy(DataCleaningStrategy):
    def __init__(self, method='remove', threshold=1.5, replace_with=None):
        self.method = method
        self.threshold = threshold
        self.replace_with = replace_with

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The data parameter must be a pandas DataFrame.")

        if self.method == 'remove':
            cleaned_data = data[(np.abs(zscore(data)) < self.threshold).all(axis=1)]
        elif self.method == 'replace':
            z_scores = np.abs(zscore(data))
            outliers = (z_scores > self.threshold)
            cleaned_data = data.copy()
            for column in data.columns:
                cleaned_data[column] = np.where(outliers[column], self.replace_with, data[column])
        elif self.method == 'ignore':
            cleaned_data = data
        else:
            raise ValueError("Invalid method. Choose from 'remove', 'replace', or 'ignore'.")

        return cleaned_data


class CustomFunctionStrategy(DataCleaningStrategy):
    def __init__(self, function):
        self.function = function

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.function(data)


class DataCleaner:
    def __init__(self, strategy: DataCleaningStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataCleaningStrategy):
        self._strategy = strategy

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame using the set strategy.

        :param data: pandas DataFrame
        :return: cleaned pandas DataFrame
        """
        return self._strategy.clean(data)

# src/data_cleaning_strategies.py

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import zscore


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
        """
        Initialize the missing values cleaning strategy.

        :param method: String, the method to use for handling missing values ('drop' or 'fill').
        :param value: Value to fill the missing entries with if method is 'fill'.
        """
        self.method = method
        self.value = value

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean missing values from the input DataFrame.

        :param data: pandas DataFrame
        :return: DataFrame with missing values handled.
        """
        if self.method == 'drop':
            cleaned_data = data.dropna()
        elif self.method == 'fill':
            cleaned_data = data.fillna(self.value)
        # ... more logic based on self.method
        return cleaned_data


class OutliersStrategy(DataCleaningStrategy):
    def __init__(self, method='remove', threshold=1.5, replace_with=None):
        """
        Initialize the outliers cleaning strategy.

        :param method: String, the method to use for handling outliers ('remove', 'replace', or 'ignore').
        :param threshold: Numeric, the threshold to use for detecting outliers.
        :param replace_with: Value to replace the outliers with if method is 'replace'.
        """
        self.method = method
        self.threshold = threshold
        self.replace_with = replace_with

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean outliers from the input DataFrame.

        :param data: pandas DataFrame
        :return: DataFrame with outliers handled.
        """
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
        """
        Initialize the custom function cleaning strategy.

        :param function: A function that takes a DataFrame and returns a cleaned DataFrame.
        """
        self.function = function

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame using a custom function, such as dropping duplicates, correcting data type

        :param data: pandas DataFrame
        :return: cleaned pandas DataFrame
        """
        return self.function(data)


class DataCleaner:
    def __init__(self, strategy: DataCleaningStrategy):
        """
        Initialize the DataCleaner with a cleaning strategy.

        :param strategy: An instance of DataCleaningStrategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataCleaningStrategy):
        """
        Set a new cleaning strategy for the DataCleaner.

        :param strategy: An instance of DataCleaningStrategy.
        """
        self._strategy = strategy

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame using the set strategy.

        :param data: pandas DataFrame
        :return: cleaned pandas DataFrame
        """
        return self._strategy.clean(data)

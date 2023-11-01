from abc import ABC, abstractmethod
from itertools import combinations

import gensim.downloader as api
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Abstract Operation Class
# All feature engineering operations will extend this class
class Operation(ABC):
    @abstractmethod
    def apply(self, context: "Context"):
        """
        Apply the operation on the data stored in context.

        :param context: The context containing the data and results.
        """
        pass


# ****************************** Context Class ******************************
# This class holds the data and the results of operations
class Context:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the context with data.

        :param data: The initial pandas DataFrame.
        """
        self.data = data
        self.results = {}

    def add_result(self, name: str, result: pd.Series):
        """
        Add a result to the context.

        :param name: Name of the result.
        :param result: The result pandas Series.
        """
        self.results[name] = result


# ****************************** Step Class ******************************
# A step consists of one or more operations
class Step:
    def __init__(self):
        """
        Initialize a step with an empty list of operations.
        """
        self.operations = []

    def add_operation(self, operation: Operation):
        """
        Add an operation to the step.

        :param operation: The operation to add.
        """
        self.operations.append(operation)

    def execute(self, context: Context):
        """
        Execute all operations in this step.

        :param context: The context to pass through the operations.
        """
        for operation in self.operations:
            operation.apply(context)


# ****************************** Pipeline Class ******************************
# A pipeline consists of one or more steps
class Pipeline:
    def __init__(self):
        """
        Initialize a pipeline with an empty list of steps.
        """
        self.steps = []

    def add_step(self, step: Step):
        """
        Add a step to the pipeline.

        :param step: The step to add.
        """
        self.steps.append(step)

    def execute(self, context: Context):
        """
        Execute all steps in this pipeline.

        :param context: The context to pass through the steps.
        """
        for step in self.steps:
            step.execute(context)


# ****************************** Specific Operations ******************************

# ============================================================================== #
# ================== 1. Creating New Features ================================== #
# ============================================================================== #
class InteractionTerms(Operation):
    def __init__(self, columns, create_for='all'):
        """
        Initialize the InteractionTerms operation.

        :param columns: List of column names to create interaction terms for.
        :param create_for: If 'all', create interaction terms for all combinations of columns.
                           If 'pairwise', create interaction terms only for consecutive pairs of columns in the list.
        """
        self.columns = columns
        self.create_for = create_for

    def apply(self, context: Context):
        """
        Apply the InteractionTerms operation to the context's data.

        :param context: The context holding the data to be transformed.
        """
        if self.create_for == 'all':
            for col1, col2 in combinations(self.columns, 2):
                context.data[f'{col1}_x_{col2}'] = context.data[col1] * context.data[col2]
        elif self.create_for == 'pairwise':
            for col1, col2 in zip(self.columns, self.columns[1:]):
                context.data[f'{col1}_x_{col2}'] = context.data[col1] * context.data[col2]


class PolynomialFeatures(Operation):
    def __init__(self, columns, degree):
        """
        Initialize the PolynomialFeatures operation.

        :param columns: List of column names for which to create polynomial features.
        :param degree: The degree of the polynomial features to be created.
        """
        self.columns = columns if isinstance(columns, list) else [columns]
        self.degree = degree

    def apply(self, context: Context):
        """
        Apply the PolynomialFeatures operation to the context's data.

        :param context: The context holding the data to be transformed.
        """
        for column in self.columns:
            for deg in range(2, self.degree + 1):
                context.data[f'{column}_pow_{deg}'] = context.data[column] ** deg


class AggregatedFeatures(Operation):
    def __init__(self, groupby_column, agg_column, agg_funcs):
        """
        Initialize the AggregatedFeatures operation.

        :param groupby_column: Column name to group by.
        :param agg_column: Column name to aggregate.
        :param agg_funcs: List of aggregation functions to apply.
        """
        self.groupby_column = groupby_column
        self.agg_column = agg_column
        self.agg_funcs = agg_funcs

    def apply(self, context: Context):
        """
        Apply the AggregatedFeatures operation to the context's data.

        :param context: The context holding the data to be transformed.
        """
        grouped_data = context.data.groupby(self.groupby_column)[self.agg_column].agg(self.agg_funcs).reset_index()
        for agg_func in self.agg_funcs:
            context.data = context.data.merge(grouped_data, on=self.groupby_column, how='left')
            context.data.rename(columns={self.agg_column: f'{self.agg_column}_{agg_func}'}, inplace=False)


class TimeBasedFeatures(Operation):
    def __init__(self, time_column):
        """
        Initialize the TimeBasedFeatures operation.

        :param time_column: Column name holding the time data.
        """
        self.time_column = time_column

    def apply(self, context: Context):
        """
        Apply the TimeBasedFeatures operation to the context's data.

        :param context: The context holding the data to be transformed.
        """
        context.data[f'{self.time_column}_day'] = context.data[self.time_column].dt.day
        context.data[f'{self.time_column}_month'] = context.data[self.time_column].dt.month
        context.data[f'{self.time_column}_year'] = context.data[self.time_column].dt.year
        context.data[f'{self.time_column}_dayofweek'] = context.data[self.time_column].dt.dayofweek
        context.data[f'{self.time_column}_hour'] = context.data[self.time_column].dt.hour


class BinFeatures(Operation):
    def __init__(self, column, n_bins):
        """
        Initialize the BinFeatures operation.

        :param column: Column name to apply binning on.
        :param n_bins: Number of bins to create.
        """
        self.column = column
        self.n_bins = n_bins

    def apply(self, context: Context):
        """
        Apply the BinFeatures operation to the context's data.

        :param context: The context holding the data to be transformed.
        """
        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal')
        context.data[f'{self.column}_binned'] = discretizer.fit_transform(context.data[[self.column]]).astype(int)


class DomainSpecificFeatures(Operation):
    def __init__(self, function):
        """
        Initialize the DomainSpecificFeatures operation.

        :param function: A function to apply to the context's data for creating domain-specific features.
        """
        self.function = function

    def apply(self, context: Context):
        """
        Apply the DomainSpecificFeatures operation to the context's data.

        :param context: The context holding the data to be transformed.
        """
        context.data = self.function(context.data)


# ============================================================================== #
# ================== 2. Encoding Categorical Variables ========================= #
# ============================================================================== #
class OneHotEncoding(Operation):
    """
    OneHotEncoding class is used to apply one-hot encoding to a specified categorical column.
    """

    def __init__(self, column):
        """
        Initializes the OneHotEncoding operation.
        :param column: The name of the column to be one-hot encoded.
        """
        self.column = column
        self.encoder = OneHotEncoder(sparse_output=False)

    def apply(self, context: Context):
        """
        Applies one-hot encoding to the specified column and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        one_hot_encoded = self.encoder.fit_transform(context.data[[self.column]])
        column_names = [f"{self.column}_{category}" for category in self.encoder.categories_[0]]
        context.data = pd.concat([context.data, pd.DataFrame(one_hot_encoded, columns=column_names)], axis=1)
        context.data.drop(columns=[self.column], inplace=True)


class LabelEncoding(Operation):
    """
    LabelEncoding class is used to apply label encoding to a specified categorical column.
    """

    def __init__(self, column):
        """
        Initializes the LabelEncoding operation.
        :param column: The name of the column to be label encoded.
        """
        self.column = column
        self.encoder = LabelEncoder()

    def apply(self, context: Context):
        """
        Applies label encoding to the specified column and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        context.data[self.column] = self.encoder.fit_transform(context.data[self.column])


class TargetEncoding(Operation):
    """
    TargetEncoding class is used to apply target encoding to a specified categorical column.
    """

    def __init__(self, column, target_column):
        """
        Initializes the TargetEncoding operation.
        :param column: The name of the categorical column to be target encoded.
        :param target_column: The name of the target column to calculate the mean for each category.
        """
        self.column = column
        self.target_column = target_column

    def apply(self, context: Context):
        """
        Applies target encoding to the specified column and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        mean_encoded = context.data.groupby(self.column)[self.target_column].mean().to_dict()
        context.data[self.column] = context.data[self.column].map(mean_encoded)


class FrequencyEncoding(Operation):
    """
    FrequencyEncoding class is used to apply frequency encoding to a specified categorical column.
    """

    def __init__(self, column):
        """
        Initializes the FrequencyEncoding operation.
        :param column: The name of the categorical column to be frequency encoded.
        """
        self.column = column

    def apply(self, context: Context):
        """
        Applies frequency encoding to the specified column and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        freq_encoded = context.data[self.column].value_counts(normalize=True).to_dict()
        context.data[self.column] = context.data[self.column].map(freq_encoded)


# ============================================================================== #
# ================== 3. Feature Transformation ================================== #
# ============================================================================== #
class LogTransformation(Operation):
    """
    LogTransformation class is used to apply a logarithmic transformation to specified numerical columns.
    """

    def __init__(self, columns):
        """
        Initializes the LogTransformation operation.
        :param columns: A list of column names to be log-transformed.
        """
        self.columns = columns

    def apply(self, context: Context):
        """
        Applies logarithmic transformation to the specified columns and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        for col in self.columns:
            context.data[col] = np.log1p(context.data[col])


class SquareRootTransformation(Operation):
    """
    SquareRootTransformation class is used to apply a square root transformation to specified numerical columns.
    """

    def __init__(self, columns):
        """
        Initializes the SquareRootTransformation operation.
        :param columns: A list of column names to be square root transformed.
        """
        self.columns = columns
        self.transformer = FunctionTransformer(np.sqrt, validate=True)

    def apply(self, context: Context):
        """
        Applies square root transformation to the specified columns and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        for col in self.columns:
            # Convert the DataFrame slice to a numpy array and apply the transformation
            transformed_array = self.transformer.transform(context.data[col].to_numpy().reshape(-1, 1)).flatten()

            # Convert the result back to a pandas Series with the correct index and name
            context.data[col] = pd.Series(transformed_array, index=context.data.index, name=col)


# ============================================================================== #
# ================== 4. Handling Missing Values ================================== #
# ============================================================================== #
class IndicatorFeatures(Operation):
    """
    IndicatorFeatures class is used to create binary features indicating whether a value is missing in another feature.
    """

    def __init__(self, columns):
        """
        Initializes the IndicatorFeatures operation.
        :param columns: A list of column names for which to create missing value indicator features.
        """
        self.columns = columns

    def apply(self, context: Context):
        """
        Applies the operation to create missing value indicator features.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        for col in self.columns:
            context.data[f'{col}_missing'] = context.data[col].isnull().astype(int)


class MissingValueImputation(Operation):
    """
    MissingValueImputation class is used to replace missing values using strategies like mean, median, mode, or model-based imputation.
    """

    def __init__(self, columns, strategy='mean'):
        """
        Initializes the MissingValueImputation operation.
        :param columns: A list of column names for which to impute missing values.
        :param strategy: The imputation strategy. One of 'mean', 'median', or 'most_frequent' (mode).
        """
        self.columns = columns
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)

    def apply(self, context: Context):
        """
        Applies the operation to impute missing values.
        :param context: The Context object containing the DataFrame to be transformed.
        """
        for col in self.columns:
            context.data[col] = self.imputer.fit_transform(context.data[col].values.reshape(-1, 1)).flatten()


# ============================================================================== #
# ================== 5. Scaling and Normalization ============================== #
# ============================================================================== #
class MinMaxScaling(Operation):
    """
    MinMaxScaling class is used to scale numerical features to a specific range, usually [0, 1].
    """

    def __init__(self, columns, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling operation.
        :param columns: A list of column names to be scaled.
        :param feature_range: Tuple (min, max) to scale the features.
        """
        self.columns = columns
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply(self, context: Context):
        """
        Applies min-max scaling to the specified columns and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be scaled.
        """
        context.data[self.columns] = self.scaler.fit_transform(context.data[self.columns])


class StandardScaling(Operation):
    """
    StandardScaling class is used to standardize numerical features by removing the mean and scaling to unit variance.
    """

    def __init__(self, columns):
        """
        Initializes the StandardScaling operation.
        :param columns: A list of column names to be standardized.
        """
        self.columns = columns
        self.scaler = StandardScaler()

    def apply(self, context: Context):
        """
        Applies standard scaling to the specified columns and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be standardized.
        """
        context.data[self.columns] = self.scaler.fit_transform(context.data[self.columns])


class RobustScaling(Operation):
    """
    RobustScaling class uses the median and the Interquartile Range for scaling, useful for datasets with outliers.
    """

    def __init__(self, columns):
        """
        Initializes the RobustScaling operation.
        :param columns: A list of column names to be robustly scaled.
        """
        self.columns = columns
        self.scaler = RobustScaler()

    def apply(self, context: Context):
        """
        Applies robust scaling to the specified columns and updates the DataFrame in the context.
        :param context: The Context object containing the DataFrame to be robustly scaled.
        """
        context.data[self.columns] = self.scaler.fit_transform(context.data[self.columns])


# ============================================================================== #
# ================== 6. Encoding Text Data ================================== #
# ============================================================================== #

class BagOfWords(Operation):
    def __init__(self, column):
        self.column = column
        self.vectorizer = CountVectorizer()

    def apply(self, context: Context):
        bow_result = self.vectorizer.fit_transform(context.data[self.column])
        context.data = pd.concat(
            [context.data, pd.DataFrame(bow_result.toarray(), columns=self.vectorizer.get_feature_names_out())], axis=1)
        context.data.drop(columns=[self.column], inplace=True)


class TFIDF(Operation):
    def __init__(self, column):
        self.column = column
        self.vectorizer = TfidfVectorizer()

    def apply(self, context: Context):
        tfidf_result = self.vectorizer.fit_transform(context.data[self.column])
        context.data = pd.concat(
            [context.data, pd.DataFrame(tfidf_result.toarray(), columns=self.vectorizer.get_feature_names_out())],
            axis=1)
        context.data.drop(columns=[self.column], inplace=True)


class WordEmbeddings(Operation):
    def __init__(self, column, model_name='glove-wiki-gigaword-50'):
        self.column = column
        self.model = api.load(model_name)

    def apply(self, context: Context):
        embeddings = []
        for text in context.data[self.column]:
            words = text.split()
            vectors = [self.model[word] for word in words if word in self.model]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(50))  # Assuming the embedding size is 50
        context.data = pd.concat([context.data, pd.DataFrame(embeddings)], axis=1)
        context.data.drop(columns=[self.column], inplace=True)


# ============================================================================== #
# ================== 7. Handling Imbalanced Data ================================== #
# ============================================================================== #
class SMOTEOperation(Operation):
    def __init__(self, target_column):
        self.target_column = target_column
        self.smote = SMOTE(random_state=42)

    def apply(self, context: Context):
        X = context.data.drop(columns=[self.target_column])
        y = context.data[self.target_column]
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        context.data = pd.concat([X_resampled, y_resampled], axis=1)


class RandomUnderSampling(Operation):
    def __init__(self, target_column):
        self.target_column = target_column
        self.under_sampler = RandomUnderSampler(random_state=42)

    def apply(self, context: Context):
        X = context.data.drop(columns=[self.target_column])
        y = context.data[self.target_column]
        X_resampled, y_resampled = self.under_sampler.fit_resample(X, y)
        context.data = pd.concat([X_resampled, y_resampled], axis=1)


class RandomOverSampling(Operation):
    def __init__(self, target_column):
        self.target_column = target_column
        self.over_sampler = RandomOverSampler(random_state=42)

    def apply(self, context: Context):
        X = context.data.drop(columns=[self.target_column])
        y = context.data[self.target_column]
        X_resampled, y_resampled = self.over_sampler.fit_resample(X, y)
        context.data = pd.concat([X_resampled, y_resampled], axis=1)


# ============================================================================== #
# ================== 8. Dimensionality Reduction ================================== #
# ============================================================================== #
class PCAOperation(Operation):
    def __init__(self, n_components):
        """
        :param n_components: Number of components to keep. If n_components is not set or None, all components are kept.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, random_state=42)

    def apply(self, context: Context):
        context.data = pd.DataFrame(self.pca.fit_transform(context.data),
                                    columns=[f"PC{i + 1}" for i in range(self.n_components)])


class FeatureSelectionOperation(Operation):
    def __init__(self, k):
        """
        :param k: Number of top features to select.
        """
        self.k = k
        self.feature_selector = SelectKBest(f_classif, k=self.k)

    def apply(self, context: Context):
        X = context.data.drop('Target', axis=1)
        y = context.data['Target']
        X_new = self.feature_selector.fit_transform(X, y)
        selected_features = context.data.columns[self.feature_selector.get_support(indices=True)]
        context.data = pd.DataFrame(X_new, columns=selected_features)


# ============================================================================== #
# ================== 9. Custom Operation ======================================= #
# ============================================================================== #

# Custom Operation
# This operation allows for the application of a custom function
class CustomOperation(Operation):
    def __init__(self, function):
        """
        Initialize the Custom Operation with a function.

        :param function: The function to apply on the data.
        """
        self.function = function

    def apply(self, context: Context):
        """
        Apply the custom function on the data.

        :param context: The context containing the data.
        """
        self.function(context)

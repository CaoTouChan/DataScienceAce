from src.feature_engineering_pipeline import *

# ============================================================================== #
# ================== 1. Creating New Features ================================== #
# ============================================================================== #

# Sample Data
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [2, 3, 2, 3, 2],
    'time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00',
                            '2023-01-04 13:00:00', '2023-01-05 14:00:00'])
})


# Sample Domain-Specific Feature Function
def create_domain_specific_features(data):
    data['D'] = data['A'] * data['C']
    return data


def test_interaction_terms():
    interaction_terms = InteractionTerms(columns=['A', 'B', 'C'], create_for='all')
    context = Context(data.copy())
    interaction_terms.apply(context)
    assert 'A_x_B' in context.data.columns
    assert 'A_x_C' in context.data.columns
    assert 'B_x_C' in context.data.columns


def test_polynomial_features():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
    })

    polynomial_features = PolynomialFeatures(columns=['A', 'B'], degree=3)
    context = Context(data.copy())
    polynomial_features.apply(context)

    # Check if polynomial features are created for both columns
    for column in ['A', 'B']:
        for degree in [2, 3]:
            generated_column_name = f'{column}_pow_{degree}'
            assert generated_column_name in context.data.columns

            # Verify the correctness of the generated polynomial features
            expected_values = np.power(data[column], degree)
            expected_values.name = generated_column_name
            pd.testing.assert_series_equal(context.data[generated_column_name], expected_values, check_names=True)


def test_aggregated_features():
    aggregated_features = AggregatedFeatures(groupby_column='C', agg_column='A', agg_funcs=['mean', 'sum'])
    context = Context(data.copy())
    aggregated_features.apply(context)
    # Update these assertions based on the actual names used in your AggregatedFeatures implementation
    assert 'mean_x' in context.data.columns
    assert 'sum_x' in context.data.columns


def test_time_based_features():
    time_based_features = TimeBasedFeatures(time_column='time')
    context = Context(data.copy())
    time_based_features.apply(context)
    assert 'time_dayofweek' in context.data.columns
    assert 'time_month' in context.data.columns
    assert 'time_hour' in context.data.columns


def test_bin_features():
    bin_features = BinFeatures(column='A', n_bins=3)
    context = Context(data.copy())
    bin_features.apply(context)
    assert 'A_binned' in context.data.columns
    assert len(context.data['A_binned'].unique()) == 3


def test_domain_specific_features():
    domain_specific_features = DomainSpecificFeatures(function=create_domain_specific_features)
    context = Context(data.copy())
    domain_specific_features.apply(context)
    assert 'D' in context.data.columns
    np.testing.assert_array_equal(context.data['D'].values, context.data['A'].values * context.data['C'].values)


# ============================================================================== #
# ================== 2. Encoding Categorical Variables ========================= #
# ============================================================================== #

def test_one_hot_encoding():
    one_hot_encoding = OneHotEncoding(column='col1')
    context = Context(pd.DataFrame({'col1': ['A', 'B', 'A']}))
    one_hot_encoding.apply(context)
    assert 'col1_A' in context.data.columns
    assert 'col1_B' in context.data.columns


def test_label_encoding():
    label_encoding = LabelEncoding(column='col1')
    context = Context(pd.DataFrame({'col1': ['A', 'B', 'A']}))
    label_encoding.apply(context)
    assert context.data['col1'].dtype == 'int'


def test_target_encoding():
    target_encoding = TargetEncoding(column='col1', target_column='target')
    context = Context(pd.DataFrame({'col1': ['A', 'B', 'A'], 'target': [1, 2, 3]}))
    target_encoding.apply(context)
    assert context.data['col1'].dtype != 'object'
    assert context.data['col1'].isna().sum() == 0


def test_frequency_encoding():
    frequency_encoding = FrequencyEncoding(column='col1')
    context = Context(pd.DataFrame({'col1': ['A', 'B', 'A']}))
    frequency_encoding.apply(context)
    assert context.data['col1'].dtype != 'object'
    assert context.data['col1'].isna().sum() == 0


# ============================================================================== #
# ================== 3. Feature Transformation ================================= #
# ============================================================================== #
def test_log_transformation():
    """
    Test the LogTransformation operation.
    """
    # Creating a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
    })

    # Applying Log Transformation to column 'A'
    context = Context(data)
    log_transformation = LogTransformation(['A'])
    log_transformation.apply(context)

    # Expected log-transformed values for column 'A'
    expected_log_A = np.log1p([1, 2, 3, 4])

    # Asserting if the operation was successful
    pd.testing.assert_series_equal(context.data['A'], pd.Series(expected_log_A, name='A'))


def test_square_root_transformation():
    """
    Test the SquareRootTransformation operation.
    """
    # Creating a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 4, 9, 16],
        'B': [5, 6, 7, 8]
    })

    # Applying Square Root Transformation to column 'A'
    context = Context(data)
    sqrt_transformation = SquareRootTransformation(['A'])
    sqrt_transformation.apply(context)

    # Expected square root transformed values for column 'A'
    expected_sqrt_A = np.sqrt([1, 4, 9, 16])

    # Asserting if the operation was successful
    pd.testing.assert_series_equal(context.data['A'], pd.Series(expected_sqrt_A, name='A'))


# ============================================================================== #
# ================== 4. Handling Missing Values ================================ #
# ============================================================================== #
def test_indicator_features():
    """
    Test the IndicatorFeatures operation.
    """
    # Creating a sample DataFrame with some missing values
    data = pd.DataFrame({
        'A': [1, None, 3, 4],
        'B': [5, 6, None, 8]
    })

    # Applying IndicatorFeatures operation to columns 'A' and 'B'
    context = Context(data)
    indicator_features = IndicatorFeatures(['A', 'B'])
    indicator_features.apply(context)

    # Expected DataFrame after adding missing value indicators
    expected_data = pd.DataFrame({
        'A': [1, None, 3, 4],
        'B': [5, 6, None, 8],
        'A_missing': [0, 1, 0, 0],
        'B_missing': [0, 0, 1, 0]
    })

    # Asserting if the operation was successful
    pd.testing.assert_frame_equal(context.data, expected_data)


def test_missing_value_imputation():
    """
    Test the MissingValueImputation operation.
    """
    # Creating a sample DataFrame with some missing values
    data = pd.DataFrame({
        'A': [1, np.nan, 3, 4],
        'B': [5, 6, np.nan, 8]
    })

    # Expected DataFrame after mean imputation
    expected_data_mean = pd.DataFrame({
        'A': [1, 2.6666666666666665, 3, 4],
        'B': [5, 6, 6.333333333333333, 8]
    })

    # Expected DataFrame after median imputation
    expected_data_median = pd.DataFrame({
        'A': [1, 3, 3, 4],
        'B': [5, 6, 6, 8]
    })

    # Expected DataFrame after most_frequent imputation
    expected_data_most_frequent = pd.DataFrame({
        'A': [1, 1, 3, 4],
        'B': [5, 6, 5, 8]
    })

    # Applying MissingValueImputation operation with mean strategy to columns 'A' and 'B'
    context_mean = Context(data.copy())
    imputation_mean = MissingValueImputation(['A', 'B'], strategy='mean')
    imputation_mean.apply(context_mean)

    # Asserting if the operation was successful for mean strategy
    pd.testing.assert_frame_equal(
        context_mean.data,
        expected_data_mean,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-8
    )

    # Applying MissingValueImputation operation with median strategy to columns 'A' and 'B'
    context_median = Context(data.copy())
    imputation_median = MissingValueImputation(['A', 'B'], strategy='median')
    imputation_median.apply(context_median)

    # Asserting if the operation was successful for median strategy
    pd.testing.assert_frame_equal(
        context_median.data,
        expected_data_median,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-8
    )

    # Applying MissingValueImputation operation with most_frequent strategy to columns 'A' and 'B'
    context_most_frequent = Context(data.copy())
    imputation_most_frequent = MissingValueImputation(['A', 'B'], strategy='most_frequent')
    imputation_most_frequent.apply(context_most_frequent)

    # Asserting if the operation was successful for most_frequent strategy
    pd.testing.assert_frame_equal(
        context_most_frequent.data,
        expected_data_most_frequent,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-8
    )


# ============================================================================== #
# ================== 5. Scaling and Normalization ============================== #
# ============================================================================== #
def test_min_max_scaling():
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    expected_data = pd.DataFrame({'A': [0, 1 / 3, 2 / 3, 1], 'B': [0, 1 / 3, 2 / 3, 1]})
    context = Context(data)
    min_max_scaling = MinMaxScaling(['A', 'B'])
    min_max_scaling.apply(context)
    pd.testing.assert_frame_equal(context.data, expected_data, check_exact=False, rtol=1e-6)


def test_standard_scaling():
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    context = Context(data)
    standard_scaling = StandardScaling(['A', 'B'])
    standard_scaling.apply(context)
    assert np.allclose(context.data.mean(), 0, atol=1e-6)
    assert np.allclose(context.data.std(ddof=0), 1, atol=1e-6)


def test_robust_scaling():
    data = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [5, 6, 7, 200]})
    context = Context(data)
    robust_scaling = RobustScaling(['A', 'B'])
    robust_scaling.apply(context)
    assert np.allclose(np.median(context.data, axis=0), 0, atol=1e-6)
    assert np.allclose(np.subtract(*np.percentile(context.data, [75, 25], axis=0)), 1, atol=1e-6)


# ============================================================================== #
# ================== 6. Encoding Text Data ===================================== #
# ============================================================================== #
def test_bag_of_words():
    data = pd.DataFrame({'Text': ['I love programming', 'Data Science is fun', 'Programming in Python']})
    context = Context(data)
    bow = BagOfWords('Text')
    bow.apply(context)
    assert 'programming' in context.data.columns
    assert 'data' in context.data.columns
    assert 'python' in context.data.columns


def test_tfidf():
    data = pd.DataFrame({'Text': ['I love programming', 'Data Science is fun', 'Programming in Python']})
    context = Context(data)
    tfidf = TFIDF('Text')
    tfidf.apply(context)
    assert 'programming' in context.data.columns
    assert 'data' in context.data.columns
    assert 'python' in context.data.columns
    assert all(context.data['programming'] >= 0)
    assert all(context.data['programming'] <= 1)


def test_word_embeddings():
    data = pd.DataFrame({'Text': ['I love programming', 'Data Science is fun', 'Programming in Python']})
    context = Context(data)
    word_embeddings = WordEmbeddings('Text')
    word_embeddings.apply(context)
    assert context.data.shape[1] == 50  # Assuming the embedding size is 50
    assert not context.data.isnull().any().any()


# ============================================================================== #
# ================== 7. Handling Imbalanced Data =============================== #
# ============================================================================== #

def test_smote():
    data = pd.DataFrame({
        'Feature': [1, 2, 3, 4],
        'Target': [0, 0, 1, 1]
    })
    context = Context(data)
    smote = SMOTEOperation('Target')
    smote.apply(context)
    assert context.data['Target'].value_counts().min() == context.data['Target'].value_counts().max()


def test_random_under_sampling():
    data = pd.DataFrame({
        'Feature': [1, 2, 3, 4],
        'Target': [0, 0, 1, 1]
    })
    context = Context(data)
    under_sampling = RandomUnderSampling('Target')
    under_sampling.apply(context)
    assert context.data['Target'].value_counts().min() == context.data['Target'].value_counts().max()


def test_random_over_sampling():
    data = pd.DataFrame({
        'Feature': [1, 2, 3, 4],
        'Target': [0, 0, 1, 1]
    })
    context = Context(data)
    over_sampling = RandomOverSampling('Target')
    over_sampling.apply(context)
    assert context.data['Target'].value_counts().min() == context.data['Target'].value_counts().max()


# ============================================================================== #
# ================== 8. Dimensionality Reduction =============================== #
# ============================================================================== #
def test_pca():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]
    })
    context = Context(data)
    pca = PCAOperation(2)
    pca.apply(context)
    assert context.data.shape[1] == 2


def test_feature_selection():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12],
        'Target': [0, 1, 0, 1]
    })
    context = Context(data)
    feature_selection = FeatureSelectionOperation(2)
    feature_selection.apply(context)
    assert context.data.shape[1] == 2
    assert 'Target' not in context.data.columns


# ============================================================================== #
# ================== 9. Custom Operation ======================================= #
# ============================================================================== #
import pandas as pd
import numpy as np

def test_custom_operation():
    # Creating a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 4, 9, 16],
        'B': [5, 6, 7, 8]
    })

    # Creating a sample function to apply to the data
    def custom_function(context):
        context.data['A'] = np.sqrt(context.data['A'])
        context.data['B'] = context.data['B'] * 10

    # Applying CustomOperation with the custom function
    context = Context(data)
    custom_operation = CustomOperation(custom_function)
    custom_operation.apply(context)

    # Expected transformed values for columns 'A' and 'B'
    expected_data = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [50, 60, 70, 80]
    })

    # Asserting if the operation was successful
    pd.testing.assert_frame_equal(context.data, expected_data)
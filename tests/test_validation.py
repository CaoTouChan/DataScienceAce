import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.validation import cross_validate_model, calculate_performance_metrics

def test_cross_validate_model():
    """
    Test the cross_validate_model function from validation.py
    """
    iris = load_iris()
    model = RandomForestClassifier(random_state=42)
    cv_results = cross_validate_model(model, iris.data, iris.target, cv=5)

    # Check that the function returns a non-empty list
    assert cv_results is not None
    assert len(cv_results) > 0

    # Check that all returned scores are between 0 and 1
    assert all(0 <= score <= 1 for score in cv_results)

def test_calculate_performance_metrics():
    """
    Test the calculate_performance_metrics function from validation.py
    """
    # Generate some dummy predictions and actuals
    predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    actuals = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])

    # Calculate performance metrics
    metrics_list = [accuracy_score]
    performance_metrics = calculate_performance_metrics(predictions, actuals, metrics_list)

    # Check that the function returns a non-empty dictionary
    assert performance_metrics is not None
    assert len(performance_metrics) > 0

    # Check that the returned metrics are correct
    assert 'accuracy_score' in performance_metrics
    assert performance_metrics['accuracy_score'] == accuracy_score(actuals, predictions)
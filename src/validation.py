from sklearn.model_selection import cross_val_score


def cross_validate_model(model, data, target, cv=5):
    """
    Perform cross-validation on the model.

    :param model: machine learning model
    :param data: data for training the model
    :param target: target values corresponding to the data
    :param cv: int, number of folds for cross-validation
    :return: cross-validation results

    Example usage:

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    model = RandomForestClassifier()
    cv_results = cross_validate_model(model, iris.data, iris.target, cv=5)
    """
    scores = cross_val_score(model, data, target, cv=cv)
    return scores


def calculate_performance_metrics(predictions, actuals, metrics_list):
    """
    Calculate performance metrics based on model predictions and actual values.

    :param predictions: model predictions
    :param actuals: actual values
    :param metrics_list: list of metric functions to calculate
    :return: calculated performance metrics

    Example usage with custom metrics:

    def mean_absolute_error(actuals, predictions):
        return sum(abs(a - p) for a, p in zip(actuals, predictions)) / len(actuals)

    metrics_list = [metrics.accuracy_score, metrics.precision_score, metrics.recall_score, metrics.f1_score, mean_absolute_error]
    performance_metrics = calculate_performance_metrics(predictions, actuals, metrics_list)
    """
    performance_metrics = {}

    for metric in metrics_list:
        metric_name = metric.__name__
        metric_value = metric(actuals, predictions)
        performance_metrics[metric_name] = metric_value

    return performance_metrics

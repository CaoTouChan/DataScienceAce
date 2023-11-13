import joblib
import pandas as pd


def save_model(model, file_path):
    """
    Save the trained model to a specified file path.

    :param model: trained machine learning model
    :param file_path: str, path to save the model
    """
    joblib.dump(model, file_path)


def load_model(file_path):
    """
    Load a trained model from a specified file path.

    :param file_path: str, path to the saved model
    :return: loaded model
    """
    model = joblib.load(file_path)
    return model


def make_prediction(model, data):
    """
    Make predictions using a trained model.

    :param model: trained machine learning model
    :param data: array-like, input data for prediction
    :return: array, model predictions
    """
    predictions = model.predict(data)
    return predictions


def save_results(results, file_path):
    """
    Save the results or performance metrics to a specified file path.

    :param results: results or performance metrics to save
    :param file_path: str, path to save the results
    """
    results_df = pd.DataFrame(results, columns=['predictions'])
    results_df.to_csv(file_path, index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def clean_data(data):
    """
    Perform cleaning operations on the raw data.

    :param data: pandas DataFrame
    :return: cleaned pandas DataFrame
    """
    # Dropping rows with missing values as an example
    # TODO: Adjust cleaning steps based on your data and requirements
    cleaned_data = data.dropna()
    print("Data cleaned successfully.")
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

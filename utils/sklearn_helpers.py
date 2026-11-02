import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data: pd.DataFrame, test_size=0.2) -> dict:
    """
    Splits the data into training and testing sets.

    Parameters:
    - data: pd.DataFrame - The input data containing features and target.
    - test_size: float - The proportion of the dataset to include in the test split.

    Returns:
    - dict: A dictionary containing the split datasets.
    """
    # Identify features and target
    features = data.columns[0:-1]
    target = data.columns[-1]

    # Extract X and y values
    X = data[features].values
    y = data[target].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=17)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
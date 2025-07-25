import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor


def get_model_classification(
    X_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def get_model_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray
) -> MLPRegressor:
    clf = MLPRegressor()
    clf.fit(X_train, y_train)
    res = MLPRegressor()  # (quantile=0.9, alpha=0.0, solver="highs")
    res.fit(X_cal, np.abs(clf.predict(X_cal) - y_cal))
    return clf, res

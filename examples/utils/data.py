from typing import Tuple

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    make_blobs,
)
from sklearn.model_selection import train_test_split


def get_data_classification(random_state: int) -> Tuple[np.ndarray, ...]:
    def generate_blobs_data(
        n_samples: int,
        n_classes: int,
        center_box: Tuple[float, float],
        random_state: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_classes,
            center_box=center_box,
            random_state=random_state,
        )
        return X, y

    X, y = generate_blobs_data(10000, 3, (-1.5, 1.5), random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test


def get_data_regression(random_state: int) -> Tuple[np.ndarray, ...]:
    X, y = fetch_california_housing(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train, y_train, test_size=0.5, random_state=random_state
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test

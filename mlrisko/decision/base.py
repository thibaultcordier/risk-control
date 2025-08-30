from abc import ABC, abstractmethod
from typing import Any, Dict, Self

import numpy as np
from sklearn.base import BaseEstimator


class BaseDecision(ABC):
    """
    Abstract base class for decision-making in risk control.

    This class provides a common interface for decision-making algorithms
    used in risk control. It includes methods for making predictions and decisions.

    # How does it work?

    - The decision-making algorithm is initialized with an estimator.
    - The estimator is used to make predictions.
    - The decision is made based on the hyper-parameters and the predictions.

    # How to use it?

    - Initialize the decision-making algorithm with an estimator
    and a parameter space.
    - Make predictions using the
    [`predict`][decision.base.BaseDecision.predict] method.
        - First, the estimator is used to make predictions
        using the [`make_prediction`][decision.base.BaseDecision.make_prediction] method.
        - Then, the decision is made based on the hyper-parameters and the predictions
        using the [`make_decision`][decision.base.BaseDecision.make_decision] method.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.

    Attributes
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.
    """  # noqa: E501

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def set_params(
        self,
        **params: Dict[str, Any],
    ) -> Self:
        """
        Set the parameters of the estimator.

        Parameters
        ----------
        **params : Dict[str, Any]
            The parameters to set on the estimator.

        Returns
        -------
        self : BaseDecision
            Returns the instance itself.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the estimator.

        Returns
        -------
        params : Dict[str, Any]
            The parameters of the estimator.
        """
        pass

    @abstractmethod
    def make_prediction(self, X: np.ndarray) -> Any:
        """
        Abstract method for predicting output values based on the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data for making predictions.

        Returns
        -------
        Any
            The predicted output values based on the input data.
        """
        pass

    @abstractmethod
    def make_decision(self, y_output: Any) -> np.ndarray:
        """
        Abstract method for predicting decisions based on output values.

        Parameters
        ----------
        y_output : Any
            The predicted output values.

        Returns
        -------
        np.ndarray
            The predicted decisions based on the output values.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict decisions based on the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data for making predictions.

        Returns
        -------
        np.ndarray
            The predicted decisions based on the input data.
        """
        y_output = self.make_prediction(X)
        y_decision = self.make_decision(y_output)
        return y_decision

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the estimator to the input data.

        This method is a placeholder and does not perform any fitting. It is
        intended to be overridden by subclasses that require fitting.

        Parameters
        ----------
        X : np.ndarray
            The input data for fitting the estimator.
        y : np.ndarray
            The target values for fitting the estimator.

        Returns
        -------
        self : BaseDecision
            Returns the instance itself.
        """
        return self

from abc import ABC
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from mlrisko.decision.base import BaseDecision


class BaseRegressionDecision(BaseDecision, ABC):
    """
    Base class for regression-based decision-making.

    This class provides a common interface for regression-based decision-making
    algorithms.

    It calls `estimator.predict(X)` to predict the output values.

    # When using this class?
    - When the decision is based on the output of a regression model.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator used to predict the output values.
    threshold : float
        The threshold for the prediction. If the prediction is less than or equal
        to the threshold, the decision is NaN; otherwise, the prediction.

    Attributes
    ----------
    estimator : BaseEstimator
        The estimator used to predict the output values.
    threshold : float
        The threshold for the residual.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        threshold: float = 0.0,
    ) -> None:
        super().__init__(estimator)
        self.threshold = threshold

    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the estimator.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        return {"threshold": self.threshold}

    def make_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output values based on the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data for making predictions.

        Returns
        -------
        np.ndarray
            The predicted output values based on the input data.
        """
        return self.estimator.predict(X)


class AdvancedRegressionDecision(BaseDecision, ABC):
    """
    Advanced class for regression-based decision-making.

    This class provides a common interface for regression-based decision-making
    algorithms based on residuals.

    It calls `estimator.predict(X)` to predict the output values and
    `residual.predict(X)` to predict the residuals.

    # When using this class?
    - When the prediction and residual are estimated separately.
    - When the decision is based on the residual.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator used to predict the output values.
    residual : BaseEstimator
        The estimator used to predict the residuals.
    threshold : float
        The threshold for the residual. If the residual is less than or equal to the
        threshold, the prediction is returned; otherwise, NaN. The default is 0.0.

    Attributes
    ----------
    estimator : BaseEstimator
        The estimator used to predict the output values.
    residual : BaseEstimator
        The estimator used to predict the residuals.
    threshold : float
        The threshold for the residual.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        residual: BaseEstimator,
        *,
        threshold: float = 1.0,
    ) -> None:
        super().__init__(estimator)
        self.residual = residual
        self.threshold = threshold

    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the estimator.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        return {"threshold": self.threshold}

    def make_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the output values and residuals for the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data for which to predict the output values and residuals.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The predicted output values and residuals.
            The first element is the predicted output values, and the second element
            is the predicted residuals.
        """
        return self.estimator.predict(X), self.residual.predict(X)

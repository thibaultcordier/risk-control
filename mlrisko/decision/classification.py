from abc import ABC
from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator

from mlrisko.decision.base import BaseDecision


class BaseClassificationDecision(BaseDecision, ABC):
    """
    Base class for classification-based decision-making.

    This class provides a common interface for classification-based decision-making
    algorithms.

    It calls `estimator.predict_proba(X)` or `estimator.decision_function(X)` to
    predict the output values.

    # When using this class?
    - When the decision is based on the output of a classification model.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.
    threshold : float
        The threshold value used for decision-making.
    predict_mode : str
        The mode to use for prediction. Can be 'proba' or 'score'.

    Attributes
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.
    threshold : float
        The threshold value used for decision-making.
    predict_mode : str
        The mode to use for prediction. Can be 'proba' or 'score'.

        - If 'proba', the `predict_proba` method of the estimator is used.
        - If 'score', the `decision_function` method of the estimator is used.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        threshold: float = 0.0,
        predict_mode: str = "score",
    ) -> None:
        super().__init__(estimator=estimator)
        self.threshold = threshold
        self.predict_mode = predict_mode

    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the estimator.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        return {
            "threshold": self.threshold,
            "predict_mode": self.predict_mode,
        }

    def make_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output values based on the input data.

        - If predict_mode is 'proba', return the predicted probabilities.
        - If predict_mode is 'score', return the decision function scores.

        Parameters
        ----------
        X : np.ndarray
            The input data for making predictions.

        Returns
        -------
        np.ndarray
            The predicted output values based on the input data.
        """
        assert self.predict_mode in [
            "proba",
            "score",
        ], "predict_mode must be 'proba' or 'score'"
        if self.predict_mode == "proba":
            return self.estimator.predict_proba(X)
        elif self.predict_mode == "score":
            return self.estimator.decision_function(X)
        else:
            return self.estimator.predict(X)

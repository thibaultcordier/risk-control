from typing import Any, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from mlrisko.abstention import _abs
from mlrisko.decision.classification import BaseClassificationDecision
from mlrisko.decision.regression import AdvancedRegressionDecision


class SelectiveRegression(AdvancedRegressionDecision):
    """
    Selective regression-based decision-making.

    1. Predict the estimated prediction and residual using two separate estimators.
    2. If the residual is less than or equal to the threshold,
        return the prediction; otherwise, NaN.

    # When using this class?
    - When the prediction and residual are estimated separately.
    - When the decision is based on the residual.
    - When you want to select predictions with low residuals.
    - When you want to deleguate the decision to a human operator when the confidence
    is low.

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

    def make_decision(self, y_output: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Predict the decision for the input data based on the output values
        and residuals.

        Parameters
        ----------
        y_output : Tuple[np.ndarray, np.ndarray]
            The output values and residuals for the input data.
            The first element is the output values, and the second element is
            the residuals

        Returns
        -------
        np.ndarray
            The predicted decision for the input data.
            The decision is made based on the output values and residuals.
            If the residual is less than or equal to the threshold, the decision is
            the output value; otherwise, the decision is NaN.
        """
        y_pred, y_res = y_output
        y_post = np.where(
            y_res <= self.threshold, y_pred, np.empty_like(y_pred) * (_abs)
        )
        return y_post


class SelectiveClassification(BaseClassificationDecision):
    """
    Selective classification-based decision-making.

    1. Predict the estimated probability or score using the estimator.
    2. If the probability or score of the top-1 class is greater than or equal to
        the threshold, return the top-1 class; otherwise, return _ABS.

    # When using this class?
    - When you want to select the top-1 class with high confidence.
    - When you want to control the false discovery rate.
    - When you want to deleguate the decision to a human operator when the
    confidence is low. So, there is abstention when the confidence is low.

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
    """

    def make_decision(self, y_output: np.ndarray) -> np.ndarray:
        """
        Predict the decision for the input data based on the output values.

        Parameters
        ----------
        y_output : np.ndarray
            The output values predicted by the estimator.

        Returns
        -------
        np.ndarray
            The predicted decisions based on the output values and threshold.
        """
        if y_output.ndim == 1:
            y_output = np.vstack([1 - y_output, y_output]).T

        y_idx = np.argmax(y_output, axis=-1)
        y_score = np.max(y_output, axis=-1)
        y_post = np.where(
            y_score >= self.threshold, y_idx, np.empty_like(y_idx) * (_abs)
        )

        return y_post


class BestClassDecision(BaseClassificationDecision):
    """
    Best classification-based decision-making.

    1. Predict the estimated probability or score using the estimator.
    2. If the probability or score of the top-1 class is greater than or equal to
        the threshold, return the top-1 class; otherwise, return empty set.

    # When using this class?
    - When you want to select the top-1 class with high confidence.
    - When you want to control the false discovery rate.
    - When you want to return empty set when the confidence is low. There is no
    abstension in this case.

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
    """

    def make_decision(self, y_output: np.ndarray) -> np.ndarray:
        """
        Predict the decision for the input data based on the output values.

        Parameters
        ----------
        y_output : np.ndarray
            The output values predicted by the estimator.

        Returns
        -------
        np.ndarray
            The predicted decisions based on the output values and threshold.
        """
        if y_output.ndim == 1:
            y_output = np.vstack([1 - y_output, y_output]).T

        n_samples, n_classes = y_output.shape

        best_score_idx = np.argmax(y_output, axis=1)
        y_post = np.zeros((n_samples, n_classes), dtype=bool)
        y_post[np.arange(n_samples), best_score_idx] = (
            y_output[np.arange(n_samples), best_score_idx] >= self.threshold
        )

        return y_post


class MultiLabelDecision(BaseClassificationDecision):
    """
    Standard multi-label classification-based decision-making.

    1. Predict the estimated probability or score using the estimator.
    2. If the probability or score of any class is greater than or equal to the
        threshold, return True; otherwise, return False.

    # When using this class?
    - When you want to select best candidates with high confidence.
    - When you want to control the false discovery rate.
    - When you want to return empty set when the confidence is low. There is no
    abstension in this case.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.
    threshold : float
        The threshold value used for decision-making.
    predict_mode : str
        The mode to use for prediction. Can be 'proba' or 'score'.
    predict_output : str
        The output type of the prediction. Can be 'class' or 'set'.

    Attributes
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.
    threshold : float
        The threshold value used for decision-making.
    predict_mode : str
        The mode to use for prediction. Can be 'proba' or 'score'.
    predict_output : str
        The output type of the prediction. Can be 'class' or 'set'.
    """

    def __init__(
        self, estimator: BaseEstimator, *, predict_output: str = "set", **kwargs: Any
    ):
        super().__init__(estimator, **kwargs)
        self.predict_output = predict_output

    def make_decision(self, y_output: np.ndarray) -> np.ndarray:
        """
        Predict the decision for the input data based on the output values.

        Parameters
        ----------
        y_output : np.ndarray
            The output values predicted by the estimator.

        Returns
        -------
        np.ndarray
            The predicted decisions based on the output values and threshold.
        """
        y_post = y_output >= self.threshold

        if self.predict_output == "class":
            return np.nanargmax(y_post, axis=-1)
        elif self.predict_output == "set":
            return y_post
        else:
            raise ValueError(f"Invalid predict_output: {self.predict_output}")


class BinaryDecision(BaseClassificationDecision):
    """
    Binary classification-based decision-making.

    1. Predict the estimated probability or score using the estimator.
    2. If the probability or score is greater than or equal to the
        threshold, return True; otherwise, return False.

    # When using this class?
    - When you want to control the precision-recall trade-off.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.
    threshold : float
        The threshold value used for decision-making.
    predict_mode : str
        The mode to use for prediction. Can be 'proba' or 'score'.
    predict_output : str
        The output type of the prediction. Can be 'class' or 'set'.

    Attributes
    ----------
    estimator : BaseEstimator
        The estimator object used for making predictions.
    threshold : float
        The threshold value used for decision-making.
    predict_mode : str
        The mode to use for prediction. Can be 'proba' or 'score'.
    predict_output : str
        The output type of the prediction. Can be 'class' or 'set'.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        predict_output: str = "class",
        **kwargs: Any,
    ):
        super().__init__(estimator, **kwargs)
        self.predict_output = predict_output

    def make_decision(self, y_output: np.ndarray) -> np.ndarray:
        """
        Predict the decision for the input data based on the output values.

        Parameters
        ----------
        y_output : np.ndarray
            The output values predicted by the estimator.

        Returns
        -------
        np.ndarray
            The predicted decisions based on the output values and threshold.
        """
        if y_output.ndim == 2:
            y_output = y_output[..., 1]

        y_post = y_output >= self.threshold

        if y_post.ndim == 1:
            y_post = np.vstack([~y_post, y_post]).T

        if self.predict_output == "class":
            return np.nanargmax(y_post, axis=-1)
        elif self.predict_output == "set":
            return y_post
        else:
            raise ValueError(f"Invalid predict_output: {self.predict_output}")

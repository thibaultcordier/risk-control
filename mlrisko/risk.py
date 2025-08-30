from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class BaseRisk(ABC):
    """
    Abstract base class for computing risks.

    This class provides methods for computing risks based on predictions
    made by an estimator or directly from predictions and true values.

    Parameters
    ----------
    acceptable_risk : float
        The acceptable risk value.

    Attributes
    ----------
    name : str
        The name of the risk function.
    greater_is_better : bool
        Whether a higher risk value is better.
    acceptable_risk : float
        The acceptable risk value.
    """

    name: str
    greater_is_better: bool

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return x

    def __init__(self, acceptable_risk: float):
        self.acceptable_risk = acceptable_risk

    def _compute_from_estimator(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute the risk based on predictions made by an estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            The estimator used to make predictions.
            Need to implement `predict` method.
        X : np.ndarray
            The input samples.
        y_true : np.ndarray
            The true values.
        **kwargs : dict
            Additional keyword arguments (used in [`compute`][risk.BaseRisk.compute]).

        Returns
        -------
        float
            The computed risk.
        """
        y_pred = estimator.predict(X)
        return self._compute_from_predictions(y_pred, y_true, **kwargs)

    def _compute_from_predictions(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute the risk based on predictions and true values.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted values.
        y_true : np.ndarray
            The true values.
        **kwargs : dict
            Additional keyword arguments (used in [`compute`][risk.BaseRisk.compute]).

        Returns
        -------
        float
            The computed risk.
        """
        return self._compute_mean(y_pred, y_true, **kwargs)

    def _compute_mean(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute the mean of the computed risks (ignoring NaNs).

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted values.
        y_true : np.ndarray
            The true values.
        **kwargs : dict
            Additional keyword arguments (used in [`compute`][risk.BaseRisk.compute]).

        Returns
        -------
        float
            The mean of the computed risks.
        """
        mean = np.nanmean(self.compute(y_pred, y_true, **kwargs))
        if mean.ndim == 0 and np.isnan(mean):
            mean = (-1 if self.greater_is_better else 1) * np.inf
        return mean

    @abstractmethod
    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the risks based on predictions and true values.

        This method should be implemented in a subclass.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted values.
        y_true : np.ndarray
            The true values.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.

        Raises
        ------
        NotImplementedError
            If this method is not implemented in a subclass.
        """
        pass


class MSERisk(BaseRisk):
    """
    A class used to compute risks based on Mean Squared Error (MSE).

    Parameters
    ----------
    mse_max : float, optional
        The maximum value for Mean Squared Error (MSE).

    Attributes
    ----------
    mse_max : float
        The maximum value for Mean Squared Error (MSE).
    """

    name: str = "mse"
    greater_is_better: bool = False

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return x * self.mse_max

    def __init__(self, acceptable_risk: float, *, mse_max: float = 1.0) -> None:
        super().__init__(acceptable_risk)
        self.mse_max = mse_max
        self.acceptable_risk = self.acceptable_risk / self.mse_max

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Computes the risks based on the predicted and true values.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted values.
        y_true : np.ndarray
            The true values.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        return np.clip((y_pred - y_true) ** 2 / self.mse_max, 0, 1)


class PrecisionRisk(BaseRisk):
    """
    A class used to compute risks based on the precision of predictions.
    """

    name: str = "precision"
    greater_is_better: bool = True

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return 1 - x

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute risks based on the precision of predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.
        y_true : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        risks = 1.0 - (y_pred == y_true)
        risks[~np.bool(y_pred)] = np.nan
        return risks


class RecallRisk(BaseRisk):
    """
    A class used to compute risks based on the recall of predictions.
    """

    name: str = "recall"
    greater_is_better: bool = True

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return 1 - x

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute risks based on the recall of predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.
        y_true : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        risks = 1.0 - (y_pred == y_true)
        risks[~np.bool(y_true)] = np.nan
        return risks


class AccuracyRisk(BaseRisk):
    """
    A class used to compute risks based on the accuracy of predictions.

    Instead of [`CoverageRisk`][risk.CoverageRisk], this class uses the best class
    prediction to compute the risk. It tests if the best class prediction is equal
    to the true label.

    It is not relevant to use this class if the decision is a prediction set
    because the best class prediction is not defined.

    At this time, no decision class uses scoring decisions, so this class
    is not used.

    - Could be relevant for [`SelectiveClassification`][decision.SelectiveClassification].
    - Irrelevant for [`MultiSelectiveClassification`][decision.MultiSelectiveClassification].
    """  # noqa: E501

    name: str = "accuracy"
    greater_is_better: bool = True

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return 1 - x

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute risks based on the accuracy of predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.
        y_true : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        if y_pred.ndim == 1:
            risks = 1.0 - (y_pred == y_true)
            risks[np.isnan(y_pred)] = np.nan
            return risks

        indexes_abs = np.any(np.isnan(y_pred), axis=-1)
        indexes_false = np.all(~np.bool(y_pred), axis=-1)
        risks = 1.0 - (np.nanargmax(y_pred, axis=-1) == y_true)
        # risks = np.where(
        #     np.all(np.isnan(y_pred), axis=-1),
        #     np.empty_like(y_true) * (_abs),
        #     1.0 - (np.nanargmax(y_pred, axis=-1) == y_true),
        # )
        risks[indexes_abs] = np.nan
        risks[indexes_false] = 1.0
        return risks


class CoverageRisk(BaseRisk):
    """
    A class used to compute risks based on the coverage of prediction sets.

    Instead of [`AccuracyRisk`][risk.AccuracyRisk], this class uses the prediction sets
    to compute the risks. It tests if the true label is in the prediction set.

    Relevant for [`MultiSelectiveClassification`][decision.MultiSelectiveClassification].
    Compatible with [`SelectiveClassification`][decision.SelectiveClassification].
    """  # noqa: E501

    name: str = "coverage"
    greater_is_better: bool = True

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return 1 - x

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute risks based on the coverage of prediction sets.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.
        y_true : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        if y_pred.ndim == 1:
            risks = 1.0 - (y_pred == y_true)
            risks[np.isnan(y_pred)] = np.nan
            return risks

        n_samples, _ = y_pred.shape
        indexes_abs = np.any(np.isnan(y_pred), axis=-1)
        risks = 1.0 - (y_pred[np.arange(n_samples), y_true])
        risks[indexes_abs] = np.nan
        return risks


class FalseDiscoveryRisk(BaseRisk):
    """
    A class used to compute risks based on the false discory rate (or coverage of prediction sets).

    TODO: Relevant for [`MultiSelectiveClassification`][decision.MultiSelectiveClassification].
    TODO: Compatible with [`SelectiveClassification`][decision.SelectiveClassification].
    """  # noqa: E501

    name: str = "FDR"
    greater_is_better: bool = False

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return x

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute risks based on the FDR of prediction sets.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.
        y_true : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        if y_pred.ndim == 1:
            risks = 1.0 - (y_pred == y_true)
            risks[np.isnan(y_pred)] = np.nan
            return risks

        # Multi classification
        # The false discovery rate is computed according to the formula:
        # fdr = 1 - (|y_pred \cap y_true| / |y_pred|)
        # where |.| is the cardinality of the set.
        elif y_true.ndim == 1:
            n_samples, _ = y_pred.shape
            indexes_abs = np.any(np.isnan(y_pred), axis=-1)
            risks = 1.0 - (y_pred[np.arange(n_samples), y_true]) / np.sum(
                y_pred, axis=-1
            )
            risks[indexes_abs] = np.nan
        else:
            indexes_abs = np.any(np.isnan(y_pred), axis=-1)
            risks = 1.0 - np.sum(y_pred * y_true, axis=-1) / np.sum(y_pred, axis=-1)
            risks[indexes_abs] = np.nan
        return risks


class AbstentionRisk(BaseRisk):
    """
    A class used to compute risks based on the ratio human/machine predictions.
    """

    name: str = "abstension"
    greater_is_better: bool = False

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return x

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute risks based on the ratio human/machine predictions.

        - Machine predictions are assumed to be not ABSTAIN.
        - Human predictions are assumed to be ABSTAIN.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.
        y_true : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        if y_pred.ndim == 1:
            return np.isnan(y_pred)
        else:
            return np.all(np.isnan(y_pred), -1)


class NonUniqueCandidateRisk(BaseRisk):
    """
    A class used to compute risks of alternative predictions.
    """

    name: str = "non_unique_candidate_risk"
    greater_is_better: bool = False

    def convert_to_performance(self, x: float) -> float:
        """
        Convert risk to performance measure.
        If the object is a risk, the performance measure is the risk.

        Parameters
        ----------
        x : float
            The risk value.

        Returns
        -------
        float
            The performance measure.
        """
        return x

    def compute(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute risks based on the alternative predictions.

        - If the prediction set is empty, the risk is 1.
        - If the prediction set has only one element, the risk is 0.
        - If the prediction set has more than one element, the risk is 1.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.
        y_true : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The computed risks.
        """
        if y_pred.ndim == 1:
            return np.isnan(y_pred)
        else:
            return 1 - (np.sum(~np.isnan(y_pred) * y_pred, axis=-1) == 1)

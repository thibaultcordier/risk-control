"""
# Selective (Binary) Classification for Human-AI Collaboration

This example shows how to use the `SelectiveClassification` and `RiskController`
classes to perform selective classification.

In which use case? When you want to assist a human decision-maker with a machine
learning model.

The goal is to provide a model that can make predictions only when it is confident
enough to do so. We will identify three different scenarios:

- The model is confident enough to make a positive feedback (i.e., the model predicts
a positive class and is confident enough to do so).
- The model is confident enough to make a negative feedback (i.e., the model predicts
a negative class and is confident enough to do so).
- The model is not confident enough to make a feedback (i.e., the model abstains
from making a prediction).
"""

import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

basedir = os.path.abspath(os.path.join(os.path.curdir, ".."))
sys.path.append(basedir)
basedir = os.path.abspath(os.path.join(os.path.curdir, "."))
sys.path.append(basedir)

import numpy as np

from mlrisko import RiskController
from mlrisko.decision.base import BaseDecision
from mlrisko.parameter import BaseParameterSpace
from mlrisko.plot import plot_p_values, plot_risk_curve
from mlrisko.risk import (
    AbstentionRisk,
    BaseRisk,
    FalseDiscoveryRisk,
)

random_state = 42
np.random.seed(42)

##################################################
# First, we load the data and train a model.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.33, random_state=random_state
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=random_state
)

# Flip randomly 10% of the labels
y_train = np.where(
    np.random.rand(y_train.shape[0]) < 0.2,
    1 - y_train,
    y_train,
)

# model = RandomForestClassifier
with warnings.catch_warnings(action="ignore"):
    model = LogisticRegression(
        penalty="l1", solver="liblinear", random_state=random_state
    )
    model.fit(X_train, y_train)

##################################################
# At this step, what are the performance of the model on the test set?
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {score:.2f}")

##################################################
# We propose to compute the confidence interval of the performance of the model.
from scipy.stats import norm


def confidence_interval(array: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    n = len(array)
    mean = np.mean(array)
    var = np.var(array)
    se = np.sqrt(var / n)
    z = norm.ppf(1 - alpha / 2)
    return mean - z * se, mean + z * se


array = np.array(y_pred == y_test)
score_ci = confidence_interval(array, alpha=0.1)
print(f"Confidence interval of the accuracy: {score_ci[0]:.2f} - {score_ci[1]:.2f}")

##################################################
# We will define a new decision rule that will be used to make decisions.

from sklearn.base import BaseEstimator

from mlrisko.abstention import _abs
from mlrisko.decision.classification import BaseClassificationDecision


class SelectiveClassification(BaseClassificationDecision):
    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        pmin: float = 0.0,
        pmax: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(estimator=estimator, **kwargs)
        self.pmin = pmin
        self.pmax = pmax

    def make_decision(self, y_output: np.ndarray) -> np.ndarray:
        """Make a decision based on the output of the model."""
        if self.predict_mode == "score":
            (n_samples,) = y_output.shape
            y_min = np.zeros_like(y_output)
            y_max = np.ones_like(y_output)
            y_empty = np.empty_like(y_output) * (_abs)
            y_post = np.where(
                y_output <= self.pmin,
                y_min,
                np.where(y_output >= self.pmax, y_max, y_empty),
            )
        else:
            (n_samples, n_features) = y_output.shape
            y_min = np.zeros_like(n_samples)
            y_max = np.ones_like(n_samples)
            y_empty = np.empty_like(n_samples) * (_abs)
            y_post = np.where(
                y_output[..., 0] <= self.pmin,
                y_min,
                np.where(y_output[..., 1] >= self.pmax, y_max, y_empty),
            )

        return y_post


##################################################
# Here, we define the decision, the risks, and the parameter space.

decision: BaseDecision = SelectiveClassification(
    estimator=model,
    predict_mode="score",
)
risks: List[BaseRisk] = [FalseDiscoveryRisk(0.1), AbstentionRisk(0.5)]
params: BaseParameterSpace = {
    "pmax": np.linspace(-2.0, 2.0, 21),
    "pmin": np.linspace(-2.0, 2.0, 21),
    # "pmax": np.linspace(.0, 1., 51),
    # "pmin": np.linspace(.0, 1., 51),
}


def lambda_to_select(l_value: Dict[str, Any]) -> bool:
    pmin = l_value["pmin"]
    pmax = l_value["pmax"]
    return pmin <= pmax


controller = RiskController(
    decision=decision,
    risks=risks,
    params=params,
    delta=0.1,
    lambda_to_select=lambda_to_select,
)

##################################################
# Now, we fit the model and plot the results. In practice, this function will be
# used to find the valid thresholds that control the risks at the given levels
# with a confidence level given by the data.
#
# A summary of the results is printed that contains the optimal threshold and the
# corresponding risks.

controller.fit(X_calib, y_calib)
controller.summary()

##################################################
# We can plot the risk curves for each risk.

plot_risk_curve(controller)

##################################################
# We can also plot the p-values for each multiple tests (parameter space).

plot_p_values(controller)

##################################################
# Finally, we can use the optimal threshold to predict on the test set and compute
# the risks. The risks are computed on the test set and converted to performance
# metrics. We can check that the risks are controlled at the given levels.

y_pred = controller.predict(X_test)
for risk in risks:
    risk_array = risk.compute(y_pred, y_test)
    ratio = risk.convert_to_performance(np.nanmean(risk_array))
    risk_array = risk_array[~np.isnan(risk_array)]

    score_ci = confidence_interval(risk_array, alpha=0.1)
    smin = risk.convert_to_performance(score_ci[1])
    smax = risk.convert_to_performance(score_ci[0])
    print(f"{risk.name}: {ratio:.2f} | {smin:.2f} - {smax:.2f} ")

"""
# Precision-Recall Trade-off for Binary Classification

This example shows how to use the `StandardClassification` and `RiskController`
classes to perform precision-recall trade-off for binary classification.
"""

import os
import sys
import warnings

basedir = os.path.abspath(os.path.join(os.path.curdir, ".."))
sys.path.append(basedir)
basedir = os.path.abspath(os.path.join(os.path.curdir, "."))
sys.path.append(basedir)

import numpy as np
from risk_control import RiskController
from risk_control.decision.base import BaseDecision
from risk_control.decision.decision import BinaryDecision
from risk_control.parameter import BaseParameterSpace
from risk_control.plot import plot_p_values, plot_risk_curve
from risk_control.risk import (
    BaseRisk,
    PrecisionRisk,
    RecallRisk,
)

random_state = 42
np.random.seed(42)

##################################################
# First, we load the data and train a model.
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_classes=2, n_samples=5000)

# Flip randomly 10% of the labels
y = np.where(
    np.random.rand(y.shape[0]) < 0.1,
    1 - y,
    y,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_state
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=random_state
)

# model = RandomForestClassifier
with warnings.catch_warnings(action="ignore"):
    model = LogisticRegression(
        penalty="l1", solver="liblinear", random_state=random_state
    )
    model.fit(X_train, y_train)

##################################################
# Here, we define the decision, the risks, and the parameter space.

decision: BaseDecision = BinaryDecision(estimator=model)
risks: list[BaseRisk] = [PrecisionRisk(0.3), RecallRisk(0.3)]
params: BaseParameterSpace = {"threshold": np.linspace(-2.0, 2.0, 101)}

controller = RiskController(
    decision=decision,
    risks=risks,
    params=params,
    delta=0.1,
)

##################################################
# Now, we fit the model and plot the results. In practice, this function will be used to find the valid
# thresholds that control the risks at the given levels with a confidence level given by the data.
#
# A summary of the results is printed that contains the optimal threshold and the corresponding risks.

controller.fit(X_calib, y_calib)
controller.summary()

##################################################
# We can plot the risk curves for each risk.

plot_risk_curve(controller)

##################################################
# We can also plot the p-values for each multiple tests (parameter space).

plot_p_values(controller)

##################################################
# Finally, we can use the optimal threshold to predict on the test set and compute the risks.
# The risks are computed on the test set and converted to performance metrics.
# We can check that the risks are controlled at the given levels.

from scipy.stats import norm


def confidence_interval(array, alpha=0.05):
    n = len(array)
    mean = np.mean(array)
    var = np.var(array)
    se = np.sqrt(var / n)
    z = norm.ppf(1 - alpha / 2)
    return mean - z * se, mean + z * se


y_pred = controller.predict(X_test)
for risk in risks:
    risk_array = risk.compute(y_pred, y_test)
    ratio = risk.convert_to_performance(np.nanmean(risk_array))
    risk_array = risk_array[~np.isnan(risk_array)]

    score_ci = confidence_interval(risk_array, alpha=0.1)
    smin = risk.convert_to_performance(score_ci[1])
    smax = risk.convert_to_performance(score_ci[0])
    print(f"{risk.name}: {ratio:.2f} | {smin:.2f} - {smax:.2f} ")

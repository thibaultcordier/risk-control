"""
# Selective Regression

This example shows how to use the `SelectiveRegression` and `RiskController` classes
to perform selective regression.
"""

import os
import sys
from typing import List

basedir = os.path.abspath(os.path.join(os.path.curdir, ".."))
sys.path.append(basedir)
basedir = os.path.abspath(os.path.join(os.path.curdir, "."))
sys.path.append(basedir)

import numpy as np
from utils.data import get_data_regression
from utils.model import get_model_regression

from risk_control import RiskController
from risk_control.decision import SelectiveRegression
from risk_control.decision.base import BaseDecision
from risk_control.parameter import BaseParameterSpace
from risk_control.plot import plot_p_values, plot_risk_curve
from risk_control.risk import AbstentionRisk, BaseRisk, MSERisk

random_state = 42
np.random.seed(random_state)

##################################################
# First, we load the data and train a model.

mse_max = 20.0

X_train, X_cal, X_test, y_train, y_cal, y_test = get_data_regression(random_state)
clf, res = get_model_regression(X_train, y_train, X_cal, y_cal)

print(f"Mean MSE: {np.nanmean((clf.predict(X_test) - y_test) ** 2):.2f}")

##################################################
# Here, we define the decision, the risks, and the parameter space.
#
# We use the `SelectiveRegression` decision, the `MSERisk` and `AbstentionRisk` risks.
#
# - The `SelectiveRegression` decision is a selective regression decision. In practice,
# it is a regression model with a threshold on the residual. If the residual is below
# the threshold, the prediction is accepted, otherwise it is rejected. The threshold
# is the parameter to tune.
# - The `MSERisk` risk is the mean squared error risk. We want the mean squared error
# to be controlled at a given level (here 0.3, TODO: report the target performance
# instead of the target risk).
# - The `AbstentionRisk` risk is the ratio prediction risk. It is the ratio of accepted
# predictions. We want the ratio of predictions to be controlled at a given level
# (here 0.2, TODO: report the target performance instead of the target risk).
#
# We want to find the valid thresholds that control the risks at the given levels
# with a confidence level (here 0.9, TODO: report the confidence level instead of
# the delta).
#
# Among the valid thresholds, we want to find the one that minimizes the mean squared
# error (beause it is the first risk in the list of risks and `control_method="lmin"`).

parameter_range = np.linspace(0.05, 5.0, 100)

decision: BaseDecision = SelectiveRegression(estimator=clf, residual=res)
risks: List[BaseRisk] = [MSERisk(0.6, mse_max=mse_max), AbstentionRisk(0.2)]
params: BaseParameterSpace = {"threshold": parameter_range}

controller = RiskController(
    decision=decision,
    risks=risks,
    params=params,
    delta=0.1,
)

##################################################
# Now, we fit the model and plot the results. In practice, this function will be used
# to find the valid thresholds that control the risks at the given levels with a
# confidence level given by the data.
#
# A summary of the results is printed that contains the optimal threshold and the
# corresponding risks.

controller.fit(X_cal, y_cal)
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
    ratio = risk.convert_to_performance(np.nanmean(risk.compute(y_pred, y_test)))
    print(f"{risk.name}: {ratio:.2f}")

print(MSERisk(mse_max)._compute_from_predictions(controller.predict(X_test), y_test))
print(MSERisk(mse_max)._compute_from_estimator(controller, X_test, y_test))

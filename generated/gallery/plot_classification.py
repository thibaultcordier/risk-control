"""
# Selective Classification

This example shows how to use the `SelectiveClassification` and `MapieRiskControl` classes to perform selective classification.
"""

import os
import sys

from risk_control.decision.base import BaseDecision
from risk_control.parameter import BaseParameterSpace

basedir = os.path.abspath(os.path.join(os.path.curdir, ".."))
sys.path.append(basedir)
basedir = os.path.abspath(os.path.join(os.path.curdir, "."))
sys.path.append(basedir)

import numpy as np
from risk_control import MapieRiskControl
from risk_control.decision import SelectiveClassification
from risk_control.plot import plot_p_values, plot_risk_curve
from risk_control.risk import AbstentionRisk, BaseRisk, CoverageRisk, FalseDiscoveryRisk
from utils.data import get_data_classification
from utils.model import get_model_classification

random_state = 42
np.random.seed(random_state)

##################################################
# First, we load the data and train a model.

X_train, X_cal, X_test, y_train, y_cal, y_test = get_data_classification(random_state)
clf = get_model_classification(X_train, y_train)

##################################################
# We can plot the data and the decision boundary.

# plot_classification(X_train, y_train)

##################################################
# Here, we define the decision, the risks, and the parameter space.
#
# We use the `SelectiveClassification` decision, the `AccuracyRisk`, `RatioPredictionRisk`
# and `CoverageRisk` risks.
#
# - The `SelectiveClassification` decision is a selective classification decision. In practice, it is a
# classification model with a threshold on the best class confidence score. If the confidence score is above
# the threshold, the prediction is accepted, otherwise it is rejected. The threshold is the parameter to tune.
# - The `CoverageRisk` risk is the coverage risk. It is the ratio of predictions containing the true label.
# We want the coverage to be controlled at a given level (here 0.5, TODO: report the target performance
# instead of the target risk). Here, the False Discovery Risk is the opposite of the coverage risk.
# - The `RatioPredictionRisk` risk is the ratio prediction risk. It is the ratio of accepted predictions.
# We want the ratio of predictions to be controlled at a given level (here 0.3, TODO: report the
# target performance instead of the target risk).
#
# We want to find the valid thresholds that control the risks at the given levels with a confidence level
# (here 0.9, TODO: report the confidence level instead of the delta).
#
# Among the valid thresholds, we want to find the one that maximizes the `AccuracyRisk` risk
# (beause it is the first risk in the list of risks and `control_method="lmin"`).

decision: BaseDecision = SelectiveClassification(estimator=clf)
risks: list[BaseRisk] = [CoverageRisk(0.2), FalseDiscoveryRisk(0.2), AbstentionRisk(0.3)]
params: BaseParameterSpace = {"threshold": np.arange(-1.0, 5.0, 0.1)}

clf_mapie = MapieRiskControl(
    decision=decision,
    risks=risks,
    params=params,
    delta=0.1,
    control_method="rmin",
)

##################################################
# Now, we fit the model and plot the results. In practice, this function will be used to find the valid
# thresholds that control the risks at the given levels with a confidence level given by the data.
#
# A summary of the results is printed that contains the optimal threshold and the corresponding risks.

clf_mapie.fit(X_cal, y_cal)
clf_mapie.summary()

##################################################
# We can plot the risk curves for each risk.

plot_risk_curve(clf_mapie)

##################################################
# We can also plot the p-values for each multiple tests (parameter space).

plot_p_values(clf_mapie)

##################################################
# Finally, we can use the optimal threshold to predict on the test set and compute the risks.
# The risks are computed on the test set and converted to performance metrics.
# We can check that the risks are controlled at the given levels.

y_pred = clf_mapie.predict(X_test)
for risk in risks:
    ratio = risk.convert_to_performance(np.nanmean(risk.compute(y_pred, y_test)))
    print(f"{risk.name}: {ratio:.2f}")

import warnings
from copy import deepcopy
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Self, Tuple, Union

import numpy as np

from mlrisko.decision import BaseDecision
from mlrisko.parameter import BaseParameterSpace
from mlrisko.risk import BaseRisk
from mlrisko.tools.fwer_control import (
    fwer_bonferroni,
    fwer_sgt,
    fwer_sgt_nd,
)
from mlrisko.tools.pvalues import compute_clt_p_values, compute_hb_p_values

# TODO: from sklearn import clone


class RiskController:
    """
    Risk control for conformal prediction.

    # Which control method to use?
    The control method consists in choosing a lambda value that controls the risk
    (defined by the user) at a given level (also defined by the user).
    Based on multiple testing, the control method gives a set of lambda values
    that control the risk. But the user has to choose one of them. And the
    strategy to choose depends on the type of risk and the type of decision.
    Here are the different strategies:

    - "lmin" : smallest lambda for which the risk is acceptable
    - "lmax" : largest lambda for which the risk is acceptable
    - "rmin" : optimal lambda for which the risk is minimized
    and acceptable
    - "rmax" : optimal lambda for which the risk is maximized
    and acceptable

    # Which pvalues methods to use?
    The pvalues method consists in computing p-values for each lambda value.
    These p-values are used to control the risk. The possible methods to
    compute p-values are:

    - "clt" : Central Limit Theorem, which is a normal approximation of the
    distribution of the risk.
    - "hb" : Hoeffding-Bentkus Inequality, which is a concentration inequality
    for the distribution of the risk.

    # Which FWER method to use?
    The FWER method (for Family-Wise Error Rate) is used to control the risk
    for multiple testing. Why? Because we have a set of lambda values, and we
    want to control the risk for all of them. They are possible methods:

    - "bonferroni" : Bonferroni Correction, which is a simple but conservative
    method that divides the significance level by the number of comparisons.
    - "sgt" : Sequential Graphical Testing (SGT), which is a more powerful method
    than Bonferroni Correction because it takes into account the space of
    hypothesis via a directed graph. The procedure sequentially tests the hypotheses
    at iteratively updayed significance levels.

    Attributes
    ----------
    decision : BaseDecision
        The decision to be made.
    params: BaseParameterSpace
        The parameter space of the decision.
        (The possible values of the lambda values).
    risks : Dict[str, BaseRisk]
        The risks to be controlled.
    delta : float
        The desired error rate (see family-wise error rate method).
    pvalue_method : str
        The method to estimate the p-values.
    fwer_method : str
        The method to control the family-wise error rate.
    control_method : str
        The method to choose the lambda value to control the risk.
    _n_samples : int
        The number of samples.
    l_values : List[dict]
        The list of lambda values (flattened parameter space).
    cr_results : dict
        A dictionary containing the risk values for each lambda value.
        The dictionary has the following structure:

        - f"risks.{key}.values": list of risk values for each lambda value.
            (for each lambda value in rows and each sample in columns)
        - f"risks.{key}.mean": list of mean risk values for each lambda value.
        - f"risks.{key}.std": list of standard deviation of risk values for each
            lambda value.
        - f"risks.{key}.pvalue": list of p-values for each lambda value.
        - "params": list of all parameters for each lambda value.
        - f"params.{key}" Additional keys for each parameter in the parameter space.
    valid_lambdas : np.ndarray
        The valid lambda values (for which the p-value is less than alpha).
    valid_risks : Dict[str, np.ndarray]
        The valid risk values (for which the p-value is less than alpha)
        (keys are the risk names).
    l_star : float
        The optimal lambda value (optimizing the risk).
    r_star : float
        The optimal risk value.
    has_solution : bool
        Whether a solution exists.
    _valid_pvalues_method : dict
        The valid p-values methods.
    _valid_fwer_method : dict
        The valid FWER methods.
    _valid_control_method : dict
        The valid control methods (defining the criteria for selecting the optimal
        lambda value).
    """

    _valid_pvalues_method: dict[str, Callable[[np.ndarray, float, int], np.ndarray]] = {  # noqa: RUF012
        "clt": compute_clt_p_values,
        "hb": compute_hb_p_values,
    }

    _valid_fwer_method: Dict[str, Callable] = {  # noqa: RUF012
        "standard": fwer_bonferroni,
        # TODO: fixed sequence testing
        "sgt_old": fwer_sgt,
        "sgt": fwer_sgt_nd,
    }

    _valid_control_method: Dict[str, Callable] = {  # noqa: RUF012
        "lmin": lambda self: np.argmin(
            [elt[self.ref_param] for elt in self.valid_lambdas]
        ),  # TODO: not working because elements are dictionary
        "lmax": lambda self: np.argmax(
            [elt[self.ref_param] for elt in self.valid_lambdas]
        ),  # TODO: not working because elements are dictionary
        "rmin": lambda self: np.argmin(self.valid_risks[self.ref_risk]),
        "rmax": lambda self: np.argmax(self.valid_risks[self.ref_risk]),
    }

    def __init__(
        self,
        decision: BaseDecision,
        params: BaseParameterSpace,
        risks: Union[BaseRisk, List[BaseRisk], Dict[str, BaseRisk]],
        *,
        delta: float,
        pvalue_method: str = "clt",
        fwer_method: str = "sgt",
        control_method: str = "rmin",
        lambda_to_select: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        """
        Initialize the RiskController class.

        Parameters
        ----------
        decision : BaseDecision
            The decision object used for making predictions and decisions.
        params : BaseParameterSpace
            The parameter space for the risk control.
        risks : Union[BaseRisk, List[BaseRisk], Dict[str, BaseRisk]]
            The risk object used for computing risk values.
        delta : float
            The desired error rate.
        pvalue_method : str
            The method to use for p-value computation ("clt" or "hb"), by default "hb".
        fwer_method : str
            The method to use for FWER control ("standard" or "sgt"), by default "sgt".
        control_method : str
            The method to use for risk control ("lmin", "lmax", "rmin", "rmax").

        Raises
        ------
        AssertionError
            If `pvalue_method`, `fwer_method` or `control_method` is not valid.
        AssertionError
            If `delta` is not in the interval (0, 1).
        """
        self.decision = decision
        self.params = params

        self.risks: Dict[str, BaseRisk]
        if isinstance(risks, list):
            self.risks = {risk_.name: risk_ for risk_ in risks}
        elif isinstance(risks, BaseRisk):
            self.risks = {risks.name: risks}
        elif isinstance(risks, dict):
            self.risks = risks

        self.target_risks: Dict[str, float] = {}
        self.target_risks = {
            risk_.name: risk_.acceptable_risk for risk_ in self.risks.values()
        }

        assert 0 < delta < 1, "delta must be in (0, 1)"
        self.delta = delta

        assert pvalue_method in self._valid_pvalues_method, "Invalid pvalue_method"
        self.pvalue_method = pvalue_method

        assert fwer_method in self._valid_fwer_method, "Invalid fwer_method"
        self.fwer_method = fwer_method

        assert control_method in self._valid_control_method, "Invalid control_method"
        self.control_method = control_method

        self.cr_results = self._initialize_cr_results()

        self.ref_risk = next(iter(self.risks.keys()))
        self.ref_param = next(iter(self.params.keys()))

        self.lambda_to_select = lambda_to_select

    def _initialize_cr_results(self) -> Dict[str, Union[List[Any], np.ndarray]]:
        """
        Initialize the control results dictionary.

        Returns
        -------
        Dict[str, Union[List[Any], np.ndarray]]
            The initialized control results dictionary.
        """
        cr_results: Dict[str, Union[List[Any], np.ndarray]] = {}

        for key in self.risks.keys():
            cr_results[f"risks.{key}.values"] = []
            cr_results[f"risks.{key}.mean"] = []
            cr_results[f"risks.{key}.std"] = []
            cr_results[f"risks.{key}.p_value"] = []
        cr_results["risks.AGG.p_value"] = []

        for key in self.params.keys():
            cr_results[f"params.{key}"] = []
        cr_results["params"] = []

        return cr_results

    def _clone_decision_with_params(
        self,
        **params: Dict[str, Any],
    ) -> BaseDecision:
        """
        Clone the decision object with the given parameters.

        Parameters
        ----------
        **params : dict
            The parameters to set on the cloned decision object.

        Returns
        -------
        BaseDecision
            The cloned decision object with the given parameters.
        """
        decision_clone: BaseDecision = deepcopy(self.decision)
        # TODO: clone(self.decision) # with scikit-learn
        decision_clone.set_params(**params)
        return decision_clone

    def _get_all_combinations(
        self, params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Tuple[int, ...]]:
        """
        Get all combinations of parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters and their possible values.

        Returns
        -------
        List[Dict[str, Any]]
            All combinations of parameters.
        Tuple[int]
            The shape of the combinations.
        """
        keys = params.keys()
        values = params.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        shape = tuple(len(v) for v in values)
        assert len(combinations) == np.prod(shape)
        return combinations, shape

    def _estimate_risk(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l_values: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Estimate the risk for each lambda value.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The true labels.
        l_values : List[Dict[str, Any]]
            The list of lambda values to evaluate.
        **kwargs : dict
            Additional keyword arguments for risk computation.

        Returns
        -------
        cr_results : Dict[str, Any]
            A dictionary containing the risk values for each lambda value.
            The dictionary has the following structure:

            - f"risks.{key}.values": list of risk values for each lambda value.
                (for each lambda value in rows and each sample in columns)
            - f"risks.{key}.mean": list of mean risk values for each lambda value.
            - f"risks.{key}.std": list of standard deviation of risk values for each
                lambda value.
            - f"risks.{key}.pvalue": list of p-values for each lambda value.
            - "params": list of all parameters for each lambda value.
            - f"params.{key}" Additional keys for each parameter in the parameter space.
        """
        cr_results: Dict[str, Any] = self._initialize_cr_results()

        y_output = self.decision.make_prediction(X)
        for l_value in l_values:
            if not (self.lambda_to_select) or self.lambda_to_select(l_value):
                new_decision = self._clone_decision_with_params(**l_value)
                y_decision = new_decision.make_decision(y_output)

                for name_, risk_ in self.risks.items():
                    risks = risk_.compute(y_decision, y, **kwargs)
                    cr_results[f"risks.{name_}.values"].append(risks.tolist())
                    cr_results[f"risks.{name_}.mean"].append(np.nanmean(risks).tolist())
                    cr_results[f"risks.{name_}.std"].append(np.nanstd(risks).tolist())

            else:
                for name_, risk_ in self.risks.items():
                    cr_results[f"risks.{name_}.values"].append(
                        [np.nan for _ in range(len(y))]
                    )
                    cr_results[f"risks.{name_}.mean"].append(np.nan)
                    cr_results[f"risks.{name_}.std"].append(np.nan)

            cr_results["params"].append(l_value)
            for key, val in l_value.items():
                cr_results[f"params.{key}"].append(val)

        # convert lists to numpy arrays for easier manipulation
        for key, val in cr_results.items():
            cr_results[key] = np.array(val)

        return cr_results

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Evaluate, for all lambda values (i.e., the grid of the decision function),
        the risk values and means for the given data with respect to the decision
        function and risk function.

        It sets the `cr_results` attribute with the results of the evaluation.
        Its a dictionary with the following structure:

        - "values": list of risk values for each lambda value.
        - "mean": list of mean risk values for each lambda value.
        - "std": list of standard deviation of risk values for each lambda value.
        - Additional keys for each parameter in the parameter space.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The true labels.
        **kwargs : dict
            Additional keyword arguments for risk estimation.

        Raises
        ------
        AssertionError
            If the number of samples in `X` and `y` do not match.
        """
        assert X.shape[0] == y.shape[0]
        self._n_samples = X.shape[0]

        self.l_values, self.param_shape = self._get_all_combinations(self.params)
        self.cr_results = self._estimate_risk(X, y, self.l_values, **kwargs)

    def _estimate_pvalues(
        self, values: np.ndarray, alpha: float, method: str
    ) -> np.ndarray:
        """
        Estimate p-values for the risk values.

        Parameters
        ----------
        values : np.ndarray
            The risk values with shape (n_params, n_samples).
        alpha : float
            The desired risk value.
        method : str
            The method to use for p-value computation ("clt" or "hb").

        Returns
        -------
        np.ndarray
            The computed p-values.

        Raises
        ------
        AssertionError
            If the method is not in the valid p-values methods.
        """
        assert method in self._valid_pvalues_method
        p_values = RiskController._valid_pvalues_method[method](
            values, alpha, self._n_samples
        )
        p_values = np.nan_to_num(p_values, nan=1.0)
        return p_values

    def _control_fwer(
        self, p_values: np.ndarray, delta: float, method: str
    ) -> np.ndarray:
        """
        Control the family-wise error rate (FWER).

        Parameters
        ----------
        p_values : np.ndarray
            The p-values with shape (n_params,).
        delta : float
            The desired error rate.
        method : str
            The method to use for FWER control ("standard" or "sgt").

        Returns
        -------
        np.ndarray
            The sorted indices of valid hypotheses.

        Warns
        -----
        UserWarning
            If no valid hypotheses are found.
        """
        indexes = RiskController._valid_fwer_method[method](
            p_values, delta, **{"param_shape": self.param_shape}
        )

        if not len(indexes):
            warnings.warn("No valid hypotheses.")
            return np.array([])
        else:
            return np.sort(indexes)

    def test(self) -> None:
        """
        Test all hypotheses and identify valid lambda values that control the
        risk and family-wise error rate. The procedure is as follows:

        1. Estimate p-values for each lambda (with
        [`_estimate_pvalues`][risk_control.RiskController._estimate_pvalues]
        method).
        2. Control the family-wise error rate (with
        [`_control_fwer`][risk_control.RiskController._control_fwer] method).
        3. Store the valid lambda values (`valid_lambdas`)
        and their corresponding risks (`valid_risks`).
        """
        for name_ in self.risks.keys():
            self.cr_results[f"risks.{name_}.p_value"] = self._estimate_pvalues(
                values=np.array(self.cr_results[f"risks.{name_}.values"]),
                alpha=self.target_risks[name_],
                method=self.pvalue_method,
            )

        self.cr_specific_results = {}
        for name_ in self.risks.keys():
            indexes = self._control_fwer(
                p_values=np.array(self.cr_results[f"risks.{name_}.p_value"]),
                delta=self.delta,
                method=self.fwer_method,
            )
            if len(indexes) > 0:
                self.cr_specific_results[name_] = {
                    "valid_lambdas": self.cr_results["params"][indexes],
                    "valid_risks": self.cr_results[f"risks.{name_}.mean"][indexes],
                    "p_values": self.cr_results[f"risks.{name_}.p_value"],
                }
            else:
                self.cr_specific_results[name_] = {
                    "valid_lambdas": [],
                    "valid_risks": [],
                    "p_values": self.cr_results[f"risks.{name_}.p_value"],
                }

        p_values = np.array(
            [self.cr_results[f"risks.{name_}.p_value"] for name_ in self.risks.keys()]
        )
        p_values = p_values.max(axis=0)
        self.cr_results["risks.AGG.p_value"] = p_values

        indexes = self._control_fwer(
            p_values=p_values,
            delta=self.delta,
            method=self.fwer_method,
        )

        self.has_solution = len(indexes) > 0

        if self.has_solution:
            self.valid_lambdas = self.cr_results["params"][indexes]
            self.valid_risks = {
                name_: self.cr_results[f"risks.{name_}.mean"][indexes]
                for name_ in self.risks.keys()
            }
        else:
            self.valid_lambdas = []
            self.valid_risks = {}

    def control(self) -> None:
        """
        Control the risk based on the specified method. The procedure is as follows:

        1. Check if a solution exists (`has_solution`).
        2. If a solution exists, select the optimal lambda value (`l_star`) and
          corresponding risk (`r_star`) based on the control method.
        3. Set the parameters of the decision model to the optimal lambda value.

        Raises
        ------
        ValueError
            If no solution is found for risk control.
        """
        self.l_star: Optional[Dict[str, float]]
        self.r_star: Optional[Dict[str, float]]

        if not self.has_solution:
            # raise ValueError("No solution found for risk control.")
            self.l_star = None
            self.r_star = None

        else:
            idx = RiskController._valid_control_method[self.control_method](self)

            self.l_star = self.valid_lambdas[idx]  # type: ignore
            self.r_star = {
                name_: valid_risks_[idx]
                for name_, valid_risks_ in self.valid_risks.items()
            }  # type: ignore
            if self.l_star:
                self.decision.set_params(**self.l_star)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> Self:
        """
        Fit the decision model on the input data, i.e.:

        1. Evaluate the decision model on the calibration data.
        2. Compute the p-values for the risk control.
        3. Find the valid lambdas for the risk control.
        4. Find the optimal lambda for the risk control.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The target labels.
        **kwargs : dict
            Additional keyword arguments to pass to the `evaluate` method.

        Returns
        -------
        self : RiskController
            The fitted risk control model.
        """
        self.evaluate(X, y, **kwargs)
        self.test()
        self.control()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the input data.

        Parameters
        ----------
        X : np.ndarray
            The input features.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        return self.decision.predict(X)

    def summary(self) -> None:
        """
        Print a summary of the risk control results.
        """
        print("=== SUMMARY ===")
        print("p(risk<=alpha) >= 1-delta")
        print(f"1-delta: {1 - self.delta:.2f}")
        print("=== risks ===")
        for name_, risk_ in self.risks.items():
            r_star = (
                risk_.convert_to_performance(self.r_star[name_])
                if self.r_star
                else np.inf
            )
            alpha = risk_.convert_to_performance(self.target_risks[name_])
            print(f"{name_}\t| optimal: {r_star:.2f}\t| alpha: {alpha:.2}")
        print("=== params ===")
        for name_ in self.params.keys():
            l_star = self.l_star[name_] if self.l_star else np.inf
            print(f"{name_}\t| optimal: {l_star:.2f}")

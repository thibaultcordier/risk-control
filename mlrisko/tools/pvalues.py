import warnings

import numpy as np
from scipy.stats import binom, norm


def compute_clt_p_values(
    risk_values: np.ndarray, alpha: float, n_samples: int
) -> np.ndarray:
    """
    Compute p-values using the Central Limit Theorem (CLT) inequality.

    Parameters
    ----------
    risk_values : np.ndarray
        Array of risk values.
    alpha : float
        Threshold value for risk.
    n_samples : int
        Number of samples used to compute risk values.

    Returns
    -------
    clt_p_values : np.ndarray
        Array of p-values computed using the CLT inequality.
    """
    with warnings.catch_warnings(action="ignore"):
        n_samples = np.count_nonzero(~np.isnan(risk_values), axis=-1)
        means = np.nanmean(risk_values, axis=-1)
        stds = np.nanstd(risk_values, axis=-1)
        clt_p_values = 1 - norm.cdf((alpha - means) / stds * np.sqrt(n_samples))

    return clt_p_values


def compute_hb_p_values(
    risk_values: np.ndarray, alpha: float, n_samples: int
) -> np.ndarray:
    """
    Compute Hoeffding-Bentkus inequality for given risk values.

    Parameters
    ----------
    risk_values : np.ndarray
        Array of risk values.
    alpha : float
        Threshold value for risk.
    n_samples : int
        Number of samples used to compute risk values.

    Returns
    -------
    hb_p_values : np.ndarray
        Array of p-values computed using the Hoeffding-Bentkus inequality.
    """
    n = np.count_nonzero(~np.isnan(risk_values), axis=-1)
    risks = np.nanmean(risk_values, axis=-1)

    def _h(r: float, a: float) -> float:
        """
        Helper function to compute Hoeffding's function.

        Parameters
        ----------
        r : float
            Risk value.
        a : float
            Significance level.

        Returns
        -------
        h : float
            Hoeffding's function value.
        """
        elt1 = r * np.log(r / a)
        elt2 = (1 - r) * np.log((1 - r) / (1 - a))
        return np.nan_to_num(elt1 + elt2)

    hoeffding_p_values = np.exp(-n * _h(np.minimum(risks, alpha), alpha))
    bentkus_p_values = np.e * binom.cdf(np.ceil(n * risks), n, alpha)
    hb_p_values = np.minimum(hoeffding_p_values, bentkus_p_values)

    return hb_p_values

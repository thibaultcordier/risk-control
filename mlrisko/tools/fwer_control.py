from typing import Any, List, Tuple

import numpy as np


def fwer_bonferroni(p_values: np.ndarray, delta: float, **kwargs: Any) -> np.ndarray:
    """
    Perform Bonferroni correction for multiple testing.

    This function adjusts the significance level (delta) for multiple
    comparisons using the Bonferroni correction method. It divides the
    significance level by the number of comparisons to control the
    family-wise error rate (FWER).

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values from individual hypothesis testing.
    delta : float
        Desired upper bound on the family-wise error rate.

    Returns
    -------
    lambda_indexes : np.ndarray
        Indices of p-values that are less than or equal to the adjusted
        significance level (delta).
    """
    new_delta = delta / len(p_values)
    lambda_indexes = np.where(p_values <= new_delta)[0]
    return lambda_indexes


def fwer_sgt(p_values: np.ndarray, delta: float, **kwargs: Any) -> np.ndarray:
    """
    Perform Sequential Graphical Testing (SGT) for multiple testing.

    This function implements the SGT procedure with FWER control, which
    sequentially tests hypotheses and adjusts the significance level to
    control the family-wise error rate (FWER).

    It works by iteratively testing the smallest p-value, rejecting the
    corresponding hypothesis if it is smaller than the current significance level,
    and adjusting the significance level for the remaining tests until no more
    hypotheses can be rejected.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values from individual hypothesis tests.
    delta : float
        Desired upper bound on the family-wise error rate.

    Returns
    -------
    lambda_array : np.ndarray
        List of indices of the hypotheses that are rejected by the SGT procedure.
    """
    # pi: list of p-values
    pi: List[float] = p_values.tolist()
    # di: list of significance levels
    di: List[float] = [delta / len(p_values)] * len(p_values)
    # lambda_indexes: list of indices of the hypotheses that are rejected
    lambda_indexes = []
    init_indexes = list(range(len(p_values)))
    idx = np.argmin(pi)

    # while the smallest p-value is smaller than the current significance level
    while pi[idx] <= di[idx]:
        # remove the hypothesis from the list of hypotheses to test
        # d = di[idx]
        # pi[idx], di[idx] = np.inf, 0
        _, d, old_idx = pi.pop(idx), di.pop(idx), init_indexes.pop(idx)

        # add the hypothesis to the list of rejected hypotheses
        lambda_indexes.append(old_idx)

        # adjust the significance levels for the remaining hypotheses
        if idx == 0:
            di[0] += d
        elif idx == len(di):
            di[-1] += d
        else:
            di[idx] += d / 2
            di[idx - 1] += d / 2

        idx = np.argmin(pi)
        np.testing.assert_approx_equal(delta, np.sum(di))

    lambda_array = np.array(lambda_indexes)

    return lambda_array


def fwer_sgt_nd(p_values: np.ndarray, delta: float, **kwargs: Any) -> np.ndarray:
    """
    Perform Sequential Graphical Testing (SGT) for multiple testing on n-dimensional
    p-values.

    This function implements the SGT procedure with FWER control for n-dimensional
    p-values. It sequentially tests hypotheses and adjusts the significance level
    to control the family-wise error rate (FWER).

    Parameters
    ----------
    p_values : np.ndarray
        n-dimensional array of p-values from individual hypothesis tests.
    delta : float
        Desired upper bound on the family-wise error rate.
    param_shape : Tuple[int, ...]
        Shape of the parameter grid.

    Returns
    -------
    lambda_array : np.ndarray
        List of indices of the hypotheses that are rejected by the SGT procedure.
    """
    param_shape: Tuple[int, ...] = kwargs.get("param_shape", p_values.shape)
    # Flatten the p-values array to a 1D list for easier manipulation
    pi = p_values.flatten().tolist()
    # Initialize the significance levels for each hypothesis
    di = [delta / p_values.size] * p_values.size
    # List to store the indices of rejected hypotheses
    lambda_indexes = []
    # Find the index of the smallest p-value
    idx = int(np.argmin(pi))

    # While the smallest p-value is smaller than the current significance level
    while pi[idx] <= di[idx]:
        # Remove the hypothesis from the list of hypotheses to test
        pi[idx] = np.inf
        d, di[idx] = di[idx], 0
        lambda_indexes.append(idx)

        # Adjust the significance levels for the remaining hypotheses
        neighbors = get_neighbors(idx, param_shape, di)
        if neighbors:
            for neighbor in neighbors:
                di[neighbor] += d / len(neighbors)
        else:
            di[np.argmin(pi)] += d

        # Find the new index of the smallest p-value
        idx = int(np.argmin(pi))
        np.testing.assert_approx_equal(delta, np.sum(di))

    # Convert the list of rejected indices to a numpy array
    lambda_array = np.array(lambda_indexes)
    return lambda_array


def get_neighbors(index: int, shape: Tuple[int, ...], di: List[float]) -> List[int]:
    """
    Get the neighbors of a given index in an n-dimensional array.

    Parameters
    ----------
    index : int
        The index of the hypothesis.
    shape : Tuple[int, ...]
        The shape of the n-dimensional array.
    di : List[float]
        The significance levels (1D array).

    Returns
    -------
    neighbors : List[int]
        List of indices (1D) of the neighboring hypotheses.
    """
    unravel_index = np.unravel_index(index, shape)
    neighbors: List[int] = []
    for dim in range(len(shape)):
        for offset in [-1, 1]:
            neighbor = list(unravel_index)
            neighbor[dim] += offset
            if 0 <= neighbor[dim] < shape[dim]:
                new_index = int(np.ravel_multi_index(neighbor, shape))
                if not (0 <= new_index < len(di)):
                    raise ValueError(
                        f"Invalid neighbor index: {new_index} when shape is {shape}, "
                        f"di.shape is {len(di)}"
                    )
                if di[new_index] != 0:
                    neighbors.append(new_index)
    return neighbors

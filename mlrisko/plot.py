import numpy as np
from matplotlib import pyplot as plt

from mlrisko.risk_control import RiskController


def plot_p_values(self: RiskController) -> None:
    """
    Plot the p-values against lambda values.
    """
    assert len(self.params) <= 2, (
        "Only one or two parameters are supported for plotting."
    )

    risk_name_ = "AGG"  # list(self.risks.keys()[0])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)

    if len(self.params) == 1:
        lambda_name = next(iter(self.params.keys()))
        lambda_values = self.cr_results[f"params.{lambda_name}"]
        p_values = self.cr_results[f"risks.{risk_name_}.p_value"]
        l_min = np.min(lambda_values)
        l_max = np.max(lambda_values)
        nb_values = len(lambda_values)

        ax.plot(lambda_values, p_values)
        ax.hlines(self.delta / nb_values, l_min, l_max, color="red")
        if self.l_star:
            l_star = self.l_star[lambda_name]
            ax.vlines(l_star, 0, 1, color="green")
        ax.set_title("lambda vs. p-value")
        plt.show()

    elif len(self.params) == 2:
        param_names = list(self.params.keys())
        lambda_values_1 = self.cr_results[f"params.{param_names[0]}"]
        lambda_values_2 = self.cr_results[f"params.{param_names[1]}"]
        p_values = self.cr_results[f"risks.{risk_name_}.p_value"]
        lambda_values_1_unique = np.unique(lambda_values_1)
        lambda_values_2_unique = np.unique(lambda_values_2)

        p_value_matrix = np.zeros(
            (len(lambda_values_1_unique), len(lambda_values_2_unique))
        )

        for i, l1 in enumerate(lambda_values_1_unique):
            for j, l2 in enumerate(lambda_values_2_unique):
                idx = np.where((lambda_values_1 == l1) & (lambda_values_2 == l2))[0]
                if idx.size > 0:
                    p_value_matrix[i, j] = p_values[idx[0]]

        ax.imshow(
            p_value_matrix,
            extent=(
                min(lambda_values_2_unique),
                max(lambda_values_2_unique),
                min(lambda_values_1_unique),
                max(lambda_values_1_unique),
            ),
            aspect="auto",
            origin="lower",
        )
        ax.set_xlabel(param_names[1])
        ax.set_ylabel(param_names[0])
        ax.set_title(f"{param_names[0]} vs. {param_names[1]} vs. p-value")
        plt.show()


def plot_risk_curve(self: RiskController) -> None:
    """
    Plot the risk curve against lambda values.
    """
    assert len(self.params) <= 2, (
        "Only one or two parameters are supported for plotting."
    )

    risk_names = list(self.risks.keys())

    if len(self.params) == 1:
        lambda_name = next(iter(self.params.keys()))
        lambda_values = self.cr_results[f"params.{lambda_name}"]

        fig, axs = plt.subplots(1, len(risk_names), figsize=(10, 5), dpi=160)
        if len(risk_names) == 1:
            axs = [axs]

        for i, risk_name in enumerate(risk_names):
            risk_means = self.cr_results[f"risks.{risk_name}.mean"]
            axs[i].plot(
                lambda_values,
                self.risks[risk_name].convert_to_performance(risk_means),  # type: ignore
                color="red",
            )
            axs[i].set_xlabel(lambda_name)
            axs[i].set_ylabel(risk_name, color="red")
            axs[i].tick_params(axis="y", labelcolor="red")

            # if True:  # len(self.valid_lambdas) > 0:
            valid_lambda = [
                elt[lambda_name]
                for elt in self.cr_specific_results[risk_name]["valid_lambdas"]
            ]
            if len(valid_lambda) > 0:
                axs[i].plot(
                    valid_lambda,
                    self.risks[risk_name].convert_to_performance(
                        self.cr_specific_results[risk_name]["valid_risks"]
                    ),
                    color="lightgreen",
                )
            lmin = np.min(lambda_values)
            lmax = np.max(lambda_values)
            axs[i].hlines(
                self.risks[risk_name].convert_to_performance(
                    self.target_risks[risk_name]
                ),
                lmin,
                lmax,
                color="red",
                linestyle="--",
            )
            if self.r_star:
                axs[i].hlines(
                    self.risks[risk_name].convert_to_performance(
                        self.r_star[risk_name]
                    ),
                    lmin,
                    lmax,
                    color="red",
                )
            if self.l_star:
                rmin = np.min(self.risks[risk_name].convert_to_performance(risk_means))  # type: ignore
                rmax = np.max(self.risks[risk_name].convert_to_performance(risk_means))  # type: ignore
                l_star = self.l_star[lambda_name]
                axs[i].vlines(l_star, rmin, rmax, color="green")
            if len(self.valid_lambdas) > 0:
                valid_lambda = [elt[lambda_name] for elt in self.valid_lambdas]
                axs[i].plot(
                    valid_lambda,
                    self.risks[risk_name].convert_to_performance(
                        self.valid_risks[risk_name]
                    ),
                    color="green",
                )

            axs[i].set_title(f"lambda vs. {risk_name}")

        fig.tight_layout(pad=2.0)
        plt.show()

    elif len(self.params) == 2:
        param_names = list(self.params.keys())
        lambda_values_1 = self.cr_results[f"params.{param_names[0]}"]
        lambda_values_2 = self.cr_results[f"params.{param_names[1]}"]

        lambda_values_1_unique = np.unique(lambda_values_1)
        lambda_values_2_unique = np.unique(lambda_values_2)

        fig, axs = plt.subplots(1, len(risk_names), figsize=(10, 5), dpi=160)
        if len(risk_names) == 1:
            axs = [axs]

        for i, risk_name in enumerate(risk_names):
            risk_means = self.cr_results[f"risks.{risk_name}.mean"]

            risk_matrix = np.zeros(
                (len(lambda_values_1_unique), len(lambda_values_2_unique))
            )

            for j, l1 in enumerate(lambda_values_1_unique):
                for k, l2 in enumerate(lambda_values_2_unique):
                    idx = np.where((lambda_values_1 == l1) & (lambda_values_2 == l2))[0]
                    if idx.size > 0:
                        risk_matrix[j, k] = self.risks[
                            risk_name
                        ].convert_to_performance(risk_means[idx[0]])

            axs[i].imshow(
                risk_matrix,
                extent=[
                    min(lambda_values_2_unique),
                    max(lambda_values_2_unique),
                    min(lambda_values_1_unique),
                    max(lambda_values_1_unique),
                ],
                aspect="auto",
                origin="lower",
            )
            axs[i].set_xlabel(param_names[1])
            axs[i].set_ylabel(param_names[0])
            axs[i].set_title(f"{param_names[0]} vs. {param_names[1]} vs. {risk_name}")

            if len(self.valid_lambdas) > 0:
                valid_lambda_1 = [elt[param_names[0]] for elt in self.valid_lambdas]
                valid_lambda_2 = [elt[param_names[1]] for elt in self.valid_lambdas]
                for l1, l2 in zip(valid_lambda_1, valid_lambda_2):
                    axs[i].plot(l2, l1, "r*", markersize=4)

        plt.show()

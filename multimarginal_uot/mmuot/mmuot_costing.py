from graph_dp import SinkhornDataProcessor
from .utils import _dual_cost_data_term, alpha_reduction
import torch
import numpy as np


def mmuot_marginal_j(
    dp: SinkhornDataProcessor, j, epsilon, prod=True, update_alpha=False
):

    marginal = torch.ones_like(dp.data_dict[j]["f"])

    # Gather incoming alphas
    for i in dp.graph.neighbors(j):
        if update_alpha:
            alpha_reduction(dp, j, i, epsilon, prod=prod)
        marginal *= dp.data_dict[(j, i)]["alpha"]

    if prod:
        marginal = (
            torch.exp(dp.data_dict[j]["f"] / epsilon)
            * marginal
            * dp.data_dict[j]["density"]
        )
    else:
        marginal = (
            torch.exp((dp.data_dict[j]["f"]) / epsilon)
            * marginal
            / np.prod(dp.data_dict[j]["f"].shape)
        )

    # marginal errror
    err = torch.linalg.norm(
        marginal.view(-1) - dp.data_dict[j]["density"].view(-1), ord=float("inf")
    ).item()

    return marginal, err


def mmuot_marginals(dp: SinkhornDataProcessor, epsilon, prod=True, alpha_update=False):

    marginals = {}
    errors = {}

    for j in dp.graph.nodes:
        marginals[j], errors[j] = mmuot_marginal_j(
            dp, j, epsilon, prod=prod, update_alpha=alpha_update
        )

    return marginals, errors


def mmuot_dual_cost(
    dp: SinkhornDataProcessor, epsilon, rho, aprox="balanced", prod=True
):
    dual_cost = 0.0

    # potential sums
    for j in dp.graph.nodes:
        dual_cost += _dual_cost_data_term(
            dp.data_dict[j]["f"], dp.data_dict[j]["density"], aprox, epsilon, rho
        )

    # entropy term
    # - epsilon < exp(sum f ) -1, K>: ToDo: do i need the <K>? cause its expensive to calculate
    marginal, _ = mmuot_marginal_j(
        dp, list(dp.graph.nodes)[0], epsilon, prod=prod, update_alpha=False
    )

    print('DUAL COST MARGINAL SUM', marginal.sum().item())

    dual_cost -= epsilon.item() * marginal.sum().item()

    return dual_cost

from graph_dp import SinkhornDataProcessor
from .utils import _dual_cost_data_term, alpha_reduction, _tensorised_alpha_reduction, alpha_reduction_pykeops
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

    # marginal error
    err = torch.linalg.norm(
        marginal.view(-1) - dp.data_dict[j]["density"].view(-1), ord=float("inf")
    ).item()

    return marginal, err


def kernal_size_reduction(dp: SinkhornDataProcessor, j, k, epsilon, prod=True):
    """
    epsilon should already be on the correct device please

    Alpha reductions, alpha_(j,k) through edges; recusive reduction

    Equation (20) in Beier et al 2022

    """

    alpha = torch.ones_like(dp.data_dict[k]["f"])
    N = np.prod(alpha.shape)

    assert alpha.shape == dp.data_dict[(k, j)]["alpha"].shape

    for i in dp.graph.neighbors(k):
        if i == j:
            continue
        else:
            # Recursivly collect alpha variables along incoming edges to node k
            alpha *= kernal_size_reduction(dp, k, i, epsilon, prod=prod)

    # Now decide how to do the reduction based on whether we are using pykeops or tensorisation
    # We only store tensorisation grid on the ordered edges... though maybe i shoudl store it on
    if "x1y1" in dp.data_dict[(j, k)] and "x2y2" in dp.data_dict[(j, k)]:
        temp = _tensorised_alpha_reduction(
            dp.data_dict[(j, k)]["x1y1"],
            dp.data_dict[(j, k)]["x2y2"],
            alpha * dp.data_dict[k]["density"] if prod else alpha / N,
            torch.zeros_like(dp.data_dict[(k)]["f"]),
            epsilon,
            dp.graph[j][k]["weight"],
        )
    elif "grid" in dp.data_dict[k] and "grid" in dp.data_dict[j]:
        temp = alpha_reduction_pykeops(
            Fi=torch.zeros_like(dp.data_dict[k]["f"]),
            Xi=dp.data_dict[k]["grid"],
            Yj=dp.data_dict[j]["grid"],
            E=epsilon,
            Mi=(
                alpha.view(-1, 1) * dp.data_dict[k]["density"].view(-1, 1)
                if prod
                else alpha.view(-1, 1) / N
            ),
            W=dp.graph[j][k]["weight"],
        )
    else:
        raise ValueError("No grid information found for alpha reduction")

    assert temp.shape == dp.data_dict[j]["f"].shape

    # ToDo: update the dictionary - this is a recusive update which will overwrite previous values
    # which is probably added a lot of updates, but if I've calcauted a new value why wouldn't I
    # keep it? I'm guessing I should - actually but for some this will be opnes!?

    return temp

def kernel_size(
    dp: SinkhornDataProcessor, epsilon, prod=True,
):
    # root node
    v0 = list(dp.graph.nodes)[0]

    kernal_red = torch.ones_like(dp.data_dict[v0]["f"])
    for neighbor in dp.graph.neighbors(v0):
        kernal_red *= kernal_size_reduction(dp, v0, neighbor, epsilon, prod=prod)
    
    if prod:
        kernal_red = (
            kernal_red
            * dp.data_dict[v0]["density"]
        )
    else:
        kernal_red = (
            kernal_red
            / np.prod(dp.data_dict[v0]["f"].shape)
        )

    return kernal_red.sum()


def mmuot_marginals(dp: SinkhornDataProcessor, epsilon, prod=True, alpha_update=False):

    marginals = {}
    errors = {}

    for j in dp.graph.nodes:
        marginals[j], errors[j] = mmuot_marginal_j(
            dp, j, epsilon, prod=prod, update_alpha=alpha_update
        )

    return marginals, errors


def mmuot_dual_cost(
    dp: SinkhornDataProcessor, epsilon, rho, aprox="balanced", prod=True, no_kernal_term=False
):
    dual_cost = 0.0

    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)

    # potential sums
    for j in dp.graph.nodes:
        # negative signs delt with within.
        dual_cost += _dual_cost_data_term(
            dp.data_dict[j]["f"], dp.data_dict[j]["density"], aprox, epsilon, rho
        )

    # entropy term
    # - epsilon < exp(sum f ) -1, K>: 
    marginal, _ = mmuot_marginal_j(
        dp, list(dp.graph.nodes)[0], epsilon, prod=prod, update_alpha=False
    )

    # k term - really to avoid recalculating the kernel size if not needed (for tests)
    if no_kernal_term:
        k_term = 0.0
    else:
        k_term = kernel_size(
            dp, epsilon, prod=True,
        ).item()

    # print('updated dual cost:', dual_cost, marginal.sum().item(), k_term)
    dual_cost -= epsilon.item() * (marginal.sum().item() - k_term)

    print('final dual cost:', dual_cost)

    return dual_cost


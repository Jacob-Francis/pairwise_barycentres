import torch
import numpy as np
from graph_dp import SinkhornDataProcessor
from .pykeops_formulas import chizat_marginals, chizat_reduction
from .utils import chizat_proxdiv_step, tensorise_f, _dual_cost_data_term, generate_epsilon_list, process_dict_for_barycentre, _tensorised_sinkhorn_reduction
from .marginals import (
    calculate_node_marginal,
    _tensorised_marginal_reduction,
    _flat_grid_marginal_reduction,
)


def asymmetric_sinkhorn_algorithm(
    data_processor: SinkhornDataProcessor,
    epsilon: float,
    rho: float,
    aprox: str,
    max_iterates: int,
    tol: float,
    epsilon_annealing: bool = False,
    debiasing: bool = True,
    verbose: bool = False,
):
    # shorten to pass around
    dp = data_processor

    process_dict_for_barycentre(dp, debiasing=debiasing)

    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)

    # Initalise the deibasing potential with barycentre shape
    d = dp._torch_numpy_process(torch.ones_like(dp.data_dict[0]["density"]))
    barycentre = d.clone() / d.sum()
    barycentre_old = d.clone() / d.sum()

    if epsilon_annealing:
        epsilon_list = generate_epsilon_list(epsilon, max_iterates)
        count_epsilon = 0
        eps = epsilon_list[count_epsilon].view(-1, 1)
        print('Epsilon annealing: Beta version, needs some tuning')
        # raise Warning("There appears to be some weird implementation errors which need fixing")
    else:
        eps = epsilon.view(-1, 1)
        count_epsilon = None

    # Initialise parameters and lists
    count_iterates = 0
    err_potentials = tol + 1.0
    err_barycentres = tol + 1.0
    potential_error_list = []
    barycentre_error_list = []

    while count_iterates < max_iterates and err_barycentres > tol:
        # reset errors
        err_potentials = -np.inf
        err_barycentres = -np.inf

        # Project edge corresponding to the data
        # I could stick these in paralell on the gpu - but for 200 by 200 I'm had problems with memory

        for edge in dp.graph.edges:
            # project on barycentre nodes edges[1]
            new_b = sinkhorn_update(dp, edge[1], edge, eps, rho, aprox, d, debiasing=debiasing)
            # calculate quasi convergnece
            err_potentials = max(
                err_potentials,
                torch.norm(new_b - dp.data_dict[edge[1]]["a"], p=float("inf")).item(),
            )
            dp.data_dict[edge[1]]["a"] = new_b

        # Barycentre updates and update barycentre in dictionary
        barycentre_old = barycentre.clone()
        barycentre = balanced_barycentre_updates(dp, d, eps)


        # calcualte error to old barycentre
        err_barycentres = torch.norm(barycentre - barycentre_old, p=float("inf")).item()

        # update the barycentre in the dictionary
        for edge in dp.graph.edges:
            dp.data_dict[edge[0]]["density"] = barycentre

        # project on second edge corresponding to the barycentre
        for edge in dp.graph.edges:
            # project on barycentre nodes edges[0]
            new_a = sinkhorn_update(dp, edge[0], edge, eps, rho, aprox="balanced", d=d, debiasing=debiasing)
            # calculate quasi convergnece
            err_potentials = max(
                err_potentials,
                torch.norm(new_a - dp.data_dict[edge[0]]["a"], p=float("inf")).item(),
            )
            dp.data_dict[edge[0]]["a"] = new_a # multiply by debiasing potential

        # Update debiasing potential
        if debiasing:
            d = debiasing_dual_potential_update(dp, d, barycentre, eps)

        # Tolerance and err_potentials or checks
        potential_error_list.append(err_potentials)
        barycentre_error_list.append(err_barycentres)

        count_iterates += 1
    
        # This will always reach the correct epsilon eventually
        # as tol is increased before reevaluting the while loop
        if epsilon_annealing:
            # converge to a lower tolerance
            if err_barycentres < tol*10 and count_epsilon < len(epsilon_list) - 1:
                count_epsilon += 1
                eps = epsilon_list[count_epsilon].view(-1, 1)
                err_barycentres = tol + 1.0  # reset to continue
                if verbose:
                    print(
                        f"Sinkhorn reached tolerance for epsilon {eps.item():.4e}, continuing to epsilon {epsilon_list[count_epsilon].item():.4e}"
                    )
            if count_epsilon == len(epsilon_list) - 1 or count_iterates > max_iterates // 2:
                # at final epsilon so turn off annealing
                # If given half the iterates switch 
                epsilon_annealing = False
                eps = epsilon.view(-1, 1)
                if verbose:
                    print('Finishing annealing at count_iterates ', count_iterates)

    if verbose:
        print(
            f"Sinkhorn finished after {count_iterates} iterations with barycentre error {err_barycentres} and potential error {err_potentials}"
        )

    if debiasing:
        # Attach potential to the graph - all pointing to the same item
        for edges in dp.graph.edges:
            dp.data_dict[edges[0]]["debiased_potential"] = d

        for edges in dp.graph.edges:
            assert (
                dp.data_dict[edges[0]]["debiased_potential"] is d
            ), "Debiasing potential should be the same object"

    return data_processor, barycentre, potential_error_list, barycentre_error_list


# def _chizat_reduction_for_sinkhorn(dp, k, edge, epsilon):
#     """

#     :param dp: Description
#     :param k: Description
#     :param edge: Description
#     :param epsilon: Description
#     """
#     assert k in edge

#     # Can I tensorise?
#     if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:
#         return _tensorised_sinkhorn_reduction(
#             dp.data_dict[k]["a"],
#             dp.data_dict[edge]["x1y1"],
#             dp.data_dict[edge]["x2y2"],
#             epsilon,
#         )
#     # Otherwise PyKeOps
#     elif "grid" in dp.data_dict[edge[0]] and "grid" in dp.data_dict[edge[1]]:

#         return _flat_grid_sinkhorn_reduction(
#             dp.data_dict[k]["a"],
#             dp.data_dict[k]["grid"],
#             dp.data_dict[edge[1] if edge[0] == k else edge[0]]["grid"],
#             epsilon,
#         )

def _chizat_reduction_for_sinkhorn(dp, k, edge, epsilon, d, debiasing=False):
    """
    Returns the reduction 
    sum_k exp((f_k - 0.5||xk - yj||^2) / epsilon) * d_{j/k}

    d is decided by debiasing flag and which node k is

    :param dp: Description
    :param k: Description
    :param edge: Description
    :param epsilon: Description
    """

    assert k in edge

    # Perform reduction to node k across edge with the kernel Kd or K.
    bary_node = edge[0]
    data_node = edge[1]

    if debiasing:
        if k == bary_node:
            a = dp.data_dict[bary_node]["a"]
            ind = 0
        elif k == data_node:
            a = dp.data_dict[data_node]["a"]
            ind = 1
        else:
            raise ValueError("k should be either bary_node or data_node")
    else:
        assert (d == torch.ones_like(dp.data_dict[bary_node]["a"])).all()
        if k == bary_node:
            a = dp.data_dict[bary_node]["a"]
            ind = 0
        elif k == data_node:
            a = dp.data_dict[data_node]["a"]
            ind = 1
        else:
            raise ValueError("k should be either bary_node or data_node")

    # checking for zeros

    # Can I tensorise?
    if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:

        temp = _tensorised_sinkhorn_reduction(
            a,
            dp.data_dict[edge]["x1y1"],
            dp.data_dict[edge]["x2y2"],
            epsilon,
            d,
            ind,
        )

    # Otherwise PyKeOps
    elif "grid" in dp.data_dict[edge[0]] and "grid" in dp.data_dict[edge[1]]:
        temp = _flat_grid_sinkhorn_reduction(
            a,
            dp.data_dict[k]["grid"],
            dp.data_dict[edge[1] if edge[0] == k else edge[0]]["grid"],
            epsilon,
            d,
            ind
        )


    if torch.any(torch.isnan(a)) or torch.any(torch.isinf(temp)):
        raise ValueError("Reduction input a has zero or negative values", a.min().item(),temp.sum().item(), k, edge)
    # if torch.any(d <= 0):
    #     raise ValueError("Reduction input d has zero or negative values", d.min().item(), temp.sum().item(),k, edge)

    # testing for NaN/inf
    if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        raise ValueError("Reduction NaN/inf detected", temp.sum().item(), a.min().item(), d.min().item(),k, edge)

    return temp


def debiasing_dual_potential_update(dp, d, barycentre, epsilon):
    """
    Debiasing requires that we know th grid for the barycentre and this may be tensorisable
    in which case we need an x1x1, x2x2 type thing in the dictionary. If the grids are the same
    then x1y1==x1x1.

    SSSSooo
    """

    # pick first edge becasue all edges should have the same barycentre node at edge[0]
    edge = list(dp.graph.edges)[0]

    # Symmetric reduction for debiasing term
    if "x1x1" in dp.data_dict[edge[0]] and "x2x2" in dp.data_dict[edge[0]]:
        s = _tensorised_sinkhorn_reduction(
            d,
            dp.data_dict[edge[0]]["x1x1"],
            dp.data_dict[edge[0]]["x2x2"],
            epsilon,
        )

    # Otherwise PyKeOps
    elif "grid" in dp.data_dict[edge[0]]:
        s = _flat_grid_sinkhorn_reduction(
            d,
            dp.data_dict[edge[0]]["grid"],
            dp.data_dict[edge[0]]["grid"],
            epsilon,
        )
    
    if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
        raise ValueError("Debiasing reduction NaN/inf detected", s.sum().item())
    
    if torch.any(s <= 0):
        raise ValueError("Debiasing reduction negative or zero values detected", s.min().item())
    
    # checking output
    output = torch.sqrt(d * barycentre / s)
    if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
        raise ValueError("Debiasing potential update NaN/inf detected", output.sum().item())
    if torch.any(output < 0):
        raise ValueError("Debiasing potential update negative or zero values detected", output.min().item())

    return torch.sqrt(d * barycentre / s)


def sinkhorn_update(dp, k, edge, epsilon, rho, aprox, d, debiasing: bool = False):
    """

    Wanted behaviour: given node k and edge (k,j) or (j,k) perform the reduction 
    to the node k. To summing against j

    old behvaiour: Given index k and edge (k,j) or (j,k) perform the reduction again the index k
    meaning the output with be 'on' node j.

    """

    assert k in edge

    #  reduction across the opposite potential
    s = _chizat_reduction_for_sinkhorn(dp, edge[1] if k == edge[0] else edge[0], edge, epsilon, d, debiasing=debiasing)
    # check the temp is valid
    if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
        raise ValueError("Sinkhorn s  update NaN/inf detected", temp.sum().item(), s.sum().item())

    temp = chizat_proxdiv_step(
        s,
        epsilon,
        rho,
        dp.data_dict[k]["density"],
        aprox=aprox,
    )

    # check the temp is valid
    if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        raise ValueError("Sinkhorn temp update NaN/inf detected", temp.sum().item(), s.sum().item(), aprox)

    return temp


def balanced_barycentre_updates(dp: SinkhornDataProcessor, d, epsilon):
    """
    I'm not sure hoe to separate this fully from the
    dictionary structure, without creating the reductions outwise the loop?
    But this would require a lot of memory. So think its better to just calcalte with the dictionary
    """

    barycentre = d.clone()
    for e1, e2, w in dp.graph.edges(data=True):
        s = _chizat_reduction_for_sinkhorn(dp, e2, (e1, e2), epsilon, d=torch.ones_like(d), debiasing=False) # because we've factored out d. 
        barycentre *= s ** w["weight"]

    # check broadcasting is correct
    assert barycentre.shape == d.shape

    return barycentre

def asymmetric_cost(
    dp: SinkhornDataProcessor,
    epsilon,
    rho,
    aprox: str,
    debiasing: bool = True,
    verbose: bool = False,
):

    epsilon = dp._torch_numpy_process(epsilon).view(-1, 1)
    rho = dp._torch_numpy_process(rho)

    us_e = []
    for edge in dp.graph.edges:
        weighting = dp.graph.edges[edge]["weight"]
        unbal_sinkhorn_div = _asymmetric_individual_edge_cost(
            dp, edge, epsilon, rho, aprox, debiasing
        )
        us_e.append(unbal_sinkhorn_div * weighting)

    if debiasing:
        # We need the last few terms
        d = dp.data_dict[edge[0]]["debiased_potential"]
        debiasing_term = _calculate_debiasing_potential_symmetric_term(
            d, dp, edge[0], epsilon
        )

        return sum(us_e) - epsilon * debiasing_term / 2, us_e
    else:
        return sum(us_e), us_e


def _asymmetric_individual_edge_cost(dp, edge, epsilon, rho, aprox, debiasing):
    bary_node = edge[0]
    data_node = edge[1]

    if debiasing:
        if "debiased_potential" in dp.data_dict[bary_node]:
            b = (
                dp.data_dict[bary_node]["a"]
                * dp.data_dict[bary_node]["debiased_potential"]
            )
            a = dp.data_dict[data_node]["a"]
        elif "debiased_potential" in dp.data_dict[data_node]:
            raise Warning("No debiasing potentials should be attached to the data")
        else:
            raise Warning(
                "No debiasing potentials attached to either node, yet using debiasing"
            )

        if (
            "debiased_potential" in dp.data_dict[bary_node]
            and "debiased_potential" in dp.data_dict[data_node]
        ):
            raise Warning(
                "Both nodes have debiasing potentials attached, this is unexpected behaviour"
            )
    else:
        a = dp.data_dict[data_node]["a"]
        b = dp.data_dict[bary_node]["a"]

    # Have sufficent information for term 1 and term 2 of dual cost
    term1 = _dual_cost_data_term(
        a, dp.data_dict[data_node]["density"], aprox, epsilon, rho
    )
    term2 = _dual_cost_data_term(
        b, dp.data_dict[bary_node]["density"], "balanced", epsilon, rho
    )
    term3 = calculate_node_marginal(dp, bary_node, epsilon, debiasing)[0].sum()

    # final constant <K>
    term4 = _calculate_dual_cost_constant(dp, edge, epsilon, debiasing)

    return term1 + term2 - epsilon * (term3 - term4)


def _calculate_dual_cost_constant(dp, edge, epsilon, debiasing):
    """
    we can hack the marginal reductions for find the cost constant summation <K>
    by using ones vectors for ai and bj
    """

    bary_node = edge[0]
    data_node = edge[1]

    if debiasing:
        if "debiased_potential" in dp.data_dict[bary_node]:
            b = (
                torch.ones_like(dp.data_dict[bary_node]["a"])
                * dp.data_dict[bary_node]["debiased_potential"]
            )
            a = torch.ones_like(dp.data_dict[data_node]["a"])
        elif "debiased_potential" in dp.data_dict[data_node]:
            raise Warning("No debiasing potentials should be attached to the data")
        else:
            raise Warning(
                "No debiasing potentials attached to either node, yet using debiasing"
            )

        if (
            "debiased_potential" in dp.data_dict[bary_node]
            and "debiased_potential" in dp.data_dict[data_node]
        ):
            raise Warning(
                "Both nodes have debiasing potentials attached, this is unexpected behaviour"
            )
    else:
        a = torch.ones_like(dp.data_dict[data_node]["a"])
        b = torch.ones_like(dp.data_dict[bary_node]["a"])

    # These terms are the same as the marginal term reductions
    if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:
        # we can tensorise
        cost_constant = _tensorised_marginal_reduction(
            dp.data_dict[edge]["x1y1"],  # either order tensorise_f will sort it
            dp.data_dict[edge]["x2y2"],
            epsilon,
            a,
            b,
        )
    elif "grid" in dp.data_dict[data_node] and "grid" in dp.data_dict[bary_node]:
        # we can use PyKeOps
        cost_constant = _flat_grid_marginal_reduction(
            dp.data_dict[data_node]["grid"],
            dp.data_dict[bary_node]["grid"],
            epsilon,
            a,
            b,
        )

    return cost_constant.sum()


def _flat_grid_sinkhorn_reduction(a, X, Y, epsilon, d=None, ind=None):

    # kernel computations - K @ a
    # main bottle neck
    if ind==0:
        return chizat_reduction(X, Y, epsilon, a*d)
    elif ind==1:
        return d*chizat_reduction(X, Y, epsilon, a)
    else:
        return chizat_reduction(X, Y, epsilon, a)

def _calculate_debiasing_potential_symmetric_term(d, dp, node, epsilon):
    """
    we can hack the marginal reductions for find the cost constant summation <K>
    by using ones vectors for ai and bj
    """

    if "x1x1" in dp.data_dict[node] and "x2x2" in dp.data_dict[node]:
        # we can tensorise
        cost_constant = _tensorised_marginal_reduction(
            dp.data_dict[node]["x1x1"],  # either order tensorise_f will sort it
            dp.data_dict[node]["x2x2"],
            epsilon,
            d - 1,
            d - 1,
        )
    elif "grid" in dp.data_dict[node]:
        # we can use PyKeOps
        cost_constant = _flat_grid_marginal_reduction(
            dp.data_dict[node]["grid"],
            dp.data_dict[node]["grid"],
            epsilon,
            d - 1,
            d - 1,
        )

    return cost_constant.sum()

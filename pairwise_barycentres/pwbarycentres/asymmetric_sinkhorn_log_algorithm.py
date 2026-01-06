import torch
import numpy as np
from graph_dp import SinkhornDataProcessor
from .pykeops_formulas import log_reduction_ii, log_reduction_ij
from .utils import (
    generate_epsilon_list,
    process_dict_for_barycentre,
    _tensorised_sinkhorn_reduction,
    _tensorised_log_sinkhorn_reduction,
    log_aprox_step,
)

from .asymmetric_sinkhorn_algorithm import (
    _flat_grid_sinkhorn_reduction,
)

def asymmetric_sinkhorn_log_algorithm(
    data_processor: SinkhornDataProcessor,
    epsilon: float,
    rho: float,
    aprox: str,
    max_iterates: int,
    tol: float,
    epsilon_annealing: bool = False,
    debiasing: bool = True,
    verbose: bool = False,
    debiasing_update_freq: int = 5,
):
    # shorten to pass around
    dp = data_processor

    # Initialise the debiasing potential with barycentre shape
    d = dp._torch_numpy_process(torch.ones_like(dp.data_dict[0]["density"]))
    barycentre = d.clone() / d.sum()
    barycentre_old = d.clone() / d.sum()

    # Preprocess dictionary for barycentre computation
    process_dict_for_barycentre(dp, debiasing=debiasing)

    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)

    if epsilon_annealing:
        epsilon_list = generate_epsilon_list(epsilon, max_iterates)
        count_epsilon = 0
        eps = epsilon_list[count_epsilon].view(-1, 1)
        print("Epsilon annealing: Beta version, needs some tuning")
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
            new_b = log_sinkhorn_update(dp, edge[1], edge, eps, rho, aprox, d=d, debiasing=True) # if False the d=None!
            if torch.any(torch.isnan(new_b)) or torch.any(torch.isinf(new_b)):
                raise ValueError("B NaN detected in sinkhorn update", new_b.sum().item(), count_iterates, edge)
            # calculate quasi convergnece
            err_potentials = max(
                err_potentials,
                torch.norm(new_b - dp.data_dict[edge[1]]["f"], p=float("inf")).item(),
            )
            dp.data_dict[edge[1]]["f"] = new_b

        # Barycentre updates and update barycentre in dictionary
        barycentre_old = barycentre.clone()
        barycentre = balanced_log_barycentre_updates(dp, d, eps, debiasing=debiasing)

        # calcualte error to old barycentre
        err_barycentres = torch.norm(barycentre - barycentre_old, p=float("inf")).item()

        # update the barycentre in the dictionary
        for edge in dp.graph.edges:
            dp.data_dict[edge[0]]["density"] = barycentre

        # project on second edge corresponding to the barycentre
        for edge in dp.graph.edges:
            # project on barycentre nodes edges[0]
            new_a = log_sinkhorn_update(dp, edge[0], edge, eps, rho, aprox="balanced", d=d, debiasing=True)
            if torch.any(torch.isnan(new_a)) or torch.any(torch.isinf(new_a)):
                raise ValueError("A NaN detected in sinkhorn update", new_a.sum().item(), count_iterates, edge)
            # calculate quasi convergnece
            err_potentials = max(
                err_potentials,
                torch.norm(new_a - dp.data_dict[edge[0]]["f"], p=float("inf")).item(),
            )
            dp.data_dict[edge[0]]["f"] = new_a

        # Update debiasing potential
        if debiasing:
            if count_iterates % debiasing_update_freq == 0:
                d = debiasing_dual_potential_update(dp, d, barycentre, eps)
                # Attach potential to the graph - all pointing to the same item
                # ToDo not have to redo this every time
                for edges in dp.graph.edges:
                    dp.data_dict[edges[0]]["debiased_potential"] = d


        # Tolerance and err_potentials or checks
        potential_error_list.append(err_potentials)
        barycentre_error_list.append(err_barycentres)

        count_iterates += 1

        # This will always reach the correct epsilon eventually
        # as tol is increased before reevaluting the while loop
        if epsilon_annealing:
            # converge to a lower tolerance
            if err_barycentres < tol * 10 and count_epsilon < len(epsilon_list) - 1:
                count_epsilon += 1
                eps = epsilon_list[count_epsilon].view(-1, 1)
                err_barycentres = tol + 1.0  # reset to continue
                if verbose:
                    print(
                        f"Sinkhorn reached tolerance for epsilon {eps.item():.4e}, continuing to epsilon {epsilon_list[count_epsilon].item():.4e}"
                    )
            if (
                count_epsilon == len(epsilon_list) - 1
                or count_iterates > max_iterates // 2
            ):
                # at final epsilon so turn off annealing
                # If given half the iterates switch
                epsilon_annealing = False
                eps = epsilon.view(-1, 1)
                if verbose:
                    print("Finishing annealing at count_iterates ", count_iterates)

    if verbose:
        print(
            f"Sinkhorn finished after {count_iterates} iterations with barycentre error {err_barycentres} and potential error {err_potentials}"
        )

    if debiasing:
         for edges in dp.graph.edges:
            assert (
                dp.data_dict[edges[0]]["debiased_potential"] is d
            ), "Debiasing potential should be the same object"

    return data_processor, barycentre, potential_error_list, barycentre_error_list


def _log_reduction_for_sinkhorn(dp, k, edge, epsilon, d=None, debiasing=True):
    """
    Returns the reduction summing over node k 
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
            f = dp.data_dict[bary_node]["f"]
            ind = 0
        elif k == data_node:
            f = dp.data_dict[data_node]["f"]
            ind = 1
        else:
            raise ValueError("k should be either bary_node or data_node")
        
        d = dp.data_dict[bary_node]["debiased_potential"]
    
        assert d is not None, "Debiasing potential should be attached to barycentre node"
    else:
        if d is None:
            d = torch.ones_like(dp.data_dict[bary_node]["f"])
        else:
            assert (d == torch.ones_like(dp.data_dict[bary_node]["f"])).all()
        if k == bary_node:
            f = dp.data_dict[bary_node]["f"]
            ind = 0
        elif k == data_node:
            f = dp.data_dict[data_node]["f"]
            ind = 1
        else:
            raise ValueError("k should be either bary_node or data_node")

    # Can I tensorise?
    if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:

        temp = _tensorised_log_sinkhorn_reduction(
            f,
            d,
            ind,
            dp.data_dict[edge]["x1y1"],
            dp.data_dict[edge]["x2y2"],
            epsilon,
        )

        # testing for NaN/inf
        if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
            raise ValueError("Tensorised reduction NaN/inf detected", temp.sum().item(), k, edge)

    # Otherwise PyKeOps
    elif "grid" in dp.data_dict[edge[0]] and "grid" in dp.data_dict[edge[1]]:
        temp = _flat_grid_log_sinkhorn_reduction(
            f,
            d,
            ind,
            dp.data_dict[k]["grid"],
            dp.data_dict[edge[1] if edge[0] == k else edge[0]]["grid"],
            epsilon,
        )

        # testing for NaN/inf
        if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):

            raise ValueError("Flat grid reduction NaN/inf detected", temp.sum().item(), k, edge)
    
    assert temp.shape == dp.data_dict[edge[0] if k == edge[1] else edge[1]]["f"].shape, "Reduction shape incorrect"
    
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

    return torch.sqrt(d * barycentre / s)

def _flat_grid_log_sinkhorn_reduction(f, d, ind, X, Y, epsilon):
    """
    if f and d are both Vi then ind= 0
    if f is Vi and d is Vj then ind=1
    """
    #something in numpy still

    # kernel computations - K @ a
    # main bottle neck
    if ind ==0:
        return log_reduction_ii(f, X, Y, epsilon, d)
    elif ind == 1:
        return log_reduction_ij(f, X, Y, epsilon, d)
    


def log_sinkhorn_update(dp, k, edge, epsilon, rho, aprox, d, debiasing):
    """

    Wanted behaviour: given node k and edge (k,j) or (j,k) perform the reduction
    to the node k. To summing against j

    old behvaiour: Given index k and edge (k,j) or (j,k) perform the reduction again the index k
    meaning the output with be 'on' node j.

    """

    assert k in edge

    #  reduction across the opposite potential
    # dp, edge[1] if k == edge[0] else edge[0], edge, epsilon, d, debiasing=debiasing
    s = epsilon*_log_reduction_for_sinkhorn(
        dp, edge[1] if k == edge[0] else edge[0], edge, epsilon, d, debiasing=debiasing
    )


    if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
        raise ValueError("log_sinkhorn_update NaN/inf detected in sinkhorn update", s.sum().item(), k, edge)


    temp = -log_aprox_step(
        s,
        epsilon,
        rho,
        dp.data_dict[k]["density"],
        aprox=aprox,
    )

    # testing aprox step
    if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        raise ValueError("log_sinkhorn_update NaN/inf detected in approx step", temp.sum().item(), k, edge)
    
    return temp


def balanced_log_barycentre_updates(dp: SinkhornDataProcessor, d, epsilon, debiasing):
    """
    I'm not sure hoe to separate this fully from the
    dictionary structure, without creating the reductions outwise the loop?
    But this would require a lot of memory. So think its better to just calcalte with the dictionary
    """

    barycentre = d.clone() # becasue we've pulled d out the front
    for e1, e2, w in dp.graph.edges(data=True):
        s = _log_reduction_for_sinkhorn(dp, e2, (e1, e2), epsilon,d=None, debiasing=False) 
        barycentre *= torch.exp(s) ** w["weight"]
    # check broadcasting is correct
    assert barycentre.shape == d.shape

    return barycentre


# def asymmetric_cost(
#     dp: SinkhornDataProcessor,
#     epsilon,
#     rho,
#     aprox: str,
#     debiasing: bool = True,
#     verbose: bool = False,
# ):

#     epsilon = dp._torch_numpy_process(epsilon).view(-1, 1)
#     rho = dp._torch_numpy_process(rho)

#     us_e = []
#     for edge in dp.graph.edges:
#         weighting = dp.graph.edges[edge]["weight"]
#         unbal_sinkhorn_div = _asymmetric_individual_edge_cost(
#             dp, edge, epsilon, rho, aprox, debiasing
#         )
#         us_e.append(unbal_sinkhorn_div * weighting)

#     if debiasing:
#         # We need the last few terms
#         d = dp.data_dict[edge[0]]["debiased_potential"]
#         debiasing_term = _calculate_debiasing_potential_symmetric_term(
#             d, dp, edge[0], epsilon
#         )

#         return sum(us_e) - epsilon * debiasing_term / 2, us_e
#     else:
#         return sum(us_e), us_e


# def _asymmetric_individual_edge_cost(dp, edge, epsilon, rho, aprox, debiasing):
#     bary_node = edge[0]
#     data_node = edge[1]

#     if debiasing:
#         if "debiased_potential" in dp.data_dict[bary_node]:
#             b = (
#                 dp.data_dict[bary_node]["a"]
#                 * dp.data_dict[bary_node]["debiased_potential"]
#             )
#             a = dp.data_dict[data_node]["a"]
#         elif "debiased_potential" in dp.data_dict[data_node]:
#             raise Warning("No debiasing potentials should be attached to the data")
#         else:
#             raise Warning(
#                 "No debiasing potentials attached to either node, yet using debiasing"
#             )

#         if (
#             "debiased_potential" in dp.data_dict[bary_node]
#             and "debiased_potential" in dp.data_dict[data_node]
#         ):
#             raise Warning(
#                 "Both nodes have debiasing potentials attached, this is unexpected behaviour"
#             )
#     else:
#         a = dp.data_dict[data_node]["a"]
#         b = dp.data_dict[bary_node]["a"]

#     # Have sufficent information for term 1 and term 2 of dual cost
#     term1 = _dual_cost_data_term(
#         a, dp.data_dict[data_node]["density"], aprox, epsilon, rho
#     )
#     term2 = _dual_cost_data_term(
#         b, dp.data_dict[bary_node]["density"], "balanced", epsilon, rho
#     )
#     term3 = calculate_node_marginal(dp, bary_node, epsilon, debiasing)[0].sum()

#     # final constant <K>
#     term4 = _calculate_dual_cost_constant(dp, edge, epsilon, debiasing)

#     return term1 + term2 - epsilon * (term3 - term4)


# def _calculate_dual_cost_constant(dp, edge, epsilon, debiasing):
#     """
#     we can hack the marginal reductions for find the cost constant summation <K>
#     by using ones vectors for ai and bj
#     """

#     bary_node = edge[0]
#     data_node = edge[1]

#     if debiasing:
#         if "debiased_potential" in dp.data_dict[bary_node]:
#             b = (
#                 torch.ones_like(dp.data_dict[bary_node]["a"])
#                 * dp.data_dict[bary_node]["debiased_potential"]
#             )
#             a = torch.ones_like(dp.data_dict[data_node]["a"])
#         elif "debiased_potential" in dp.data_dict[data_node]:
#             raise Warning("No debiasing potentials should be attached to the data")
#         else:
#             raise Warning(
#                 "No debiasing potentials attached to either node, yet using debiasing"
#             )

#         if (
#             "debiased_potential" in dp.data_dict[bary_node]
#             and "debiased_potential" in dp.data_dict[data_node]
#         ):
#             raise Warning(
#                 "Both nodes have debiasing potentials attached, this is unexpected behaviour"
#             )
#     else:
#         a = torch.ones_like(dp.data_dict[data_node]["a"])
#         b = torch.ones_like(dp.data_dict[bary_node]["a"])

#     # These terms are the same as the marginal term reductions
#     if "x1y1" in dp.data_dict[edge] and "x2y2" in dp.data_dict[edge]:
#         # we can tensorise
#         cost_constant = _tensorised_marginal_reduction(
#             dp.data_dict[edge]["x1y1"],  # either order tensorise_f will sort it
#             dp.data_dict[edge]["x2y2"],
#             epsilon,
#             a,
#             b,
#         )
#     elif "grid" in dp.data_dict[data_node] and "grid" in dp.data_dict[bary_node]:
#         # we can use PyKeOps
#         cost_constant = _flat_grid_marginal_reduction(
#             dp.data_dict[data_node]["grid"],
#             dp.data_dict[bary_node]["grid"],
#             epsilon,
#             a,
#             b,
#         )

#     return cost_constant.sum()


# def _calculate_debiasing_potential_symmetric_term(d, dp, node, epsilon):
#     """
#     we can hack the marginal reductions for find the cost constant summation <K>
#     by using ones vectors for ai and bj
#     """

#     if "x1x1" in dp.data_dict[node] and "x2x2" in dp.data_dict[node]:
#         # we can tensorise
#         cost_constant = _tensorised_marginal_reduction(
#             dp.data_dict[node]["x1x1"],  # either order tensorise_f will sort it
#             dp.data_dict[node]["x2x2"],
#             epsilon,
#             d - 1,
#             d - 1,
#         )
#     elif "grid" in dp.data_dict[node]:
#         # we can use PyKeOps
#         cost_constant = _flat_grid_marginal_reduction(
#             dp.data_dict[node]["grid"],
#             dp.data_dict[node]["grid"],
#             epsilon,
#             d - 1,
#             d - 1,
#         )

#     return cost_constant.sum()

import torch
import numpy as np
from graph_dp import SinkhornDataProcessor
from ..common.pykeops_formulas import log_reduction_ii, log_reduction_ij
from ..common.utils import (
    generate_epsilon_list,
    process_dict_for_barycentre,
    _tensorised_sinkhorn_reduction,
    _tensorised_log_sinkhorn_reduction,
    _flat_grid_sinkhorn_reduction,
    log_aprox_step,
    _flat_grid_log_sinkhorn_reduction,
    _tensorised_log_sinkhorn_reduction_stabalised
)

from ..common.marginals import calculate_node_marginal

from .asymmetric_cost import asymmetric_cost


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
    measure_constraints=False,
    termination_criterion="barycentre",
    lags=None,
    energy_tracking=False,
    fixed_barycentre=None
):
    # shorten to pass around
    dp = data_processor

    # Initialise the debiasing potential with barycentre shape
    d = dp._torch_numpy_process(torch.ones_like(dp.data_dict[0]["density"]))
    if fixed_barycentre is None:
        barycentre = d.clone() / d.sum()
        barycentre_old = d.clone() / d.sum()
    else:
        debiasing = False
        print('For a fixed barycentre we turn off debiasing')
        barycentre = dp._torch_numpy_process(fixed_barycentre).reshape(*d.shape)
        barycentre_old = dp._torch_numpy_process(fixed_barycentre).reshape(*d.shape)
        # update the barycentre in the dictionary
        for edge in dp.graph.edges:
            print('Updating barycentre in dictionary for fixed barycentre case', edge[0])
            dp.data_dict[edge[0]]["density"] = barycentre

    # Preprocess dictionary for barycentre computation
    process_dict_for_barycentre(dp, debiasing=debiasing)

    if lags is None:
        lags = {
            'barycentre': 1,
            'debiasing': 1
        }

    debiasing_update_freq = lags['debiasing']

    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)

    if epsilon_annealing:
        max_iterates = max_iterates * 3 # give more iterates to anneal
        epsilon_list = generate_epsilon_list(epsilon, max_iterates//3)
        count_epsilon = 0
        eps = epsilon_list[count_epsilon].view(-1, 1)
        print("Epsilon annealing: Beta version, needs some tuning. max_its increased x 3")
        # raise Warning("There appears to be some weird implementation errors which need fixing")
    else:
        eps = epsilon.view(-1, 1)
        count_epsilon = None

    # Initialise parameters and lists
    count_iterates = 0
    err_potentials = tol + 1.0
    err_barycentres = tol + 1.0
    err = tol + 1.0

    if measure_constraints:
        constraints_dict = {
            'partial_bary': [],
            'partial_f': [],
            'partial_d': [],
            'partial_g': [],
        }
    
    if energy_tracking:
        energy_dict = {
            'total_cost': [],
            'unbalanced_sinkhorn_terms': [],
            'debiasing_term': [],
            'uot_mu_mu_terms': []
        }

    
    potential_error_list = []
    barycentre_error_list = []

    while count_iterates < max_iterates and err > tol:
        # reset errors
        err_potentials = -np.inf
        err_barycentres = -np.inf

        # Project edge corresponding to the data
        # I could stick these in paralell on the gpu - but for 200 by 200 I'm had problems with memory

        for edge in dp.graph.edges:
            # project on barycentre nodes edges[1]
            new_b = log_sinkhorn_update(dp, edge[1], edge, eps, rho, aprox, d=None, debiasing=True) # if False the d=None!
            if torch.any(torch.isnan(new_b)) or torch.any(torch.isinf(new_b)):
                raise ValueError("B NaN detected in sinkhorn update", new_b.sum().item(), count_iterates, edge, eps)
            # calculate quasi convergnece
            err_potentials = max(
                err_potentials,
                torch.norm(new_b - dp.data_dict[edge[1]]["f"], p=float("inf")).item(),
            )
            dp.data_dict[edge[1]]["f"] = new_b

        if fixed_barycentre is None:
            # Barycentre updates and update barycentre in dictionary
            if count_iterates % lags['barycentre'] == 0 and type(lags['barycentre'])==int:
                barycentre_old = barycentre.clone()
                barycentre = balanced_log_barycentre_updates(dp, d, eps, debiasing=debiasing)

                # calcualte error to old barycentre
                err_barycentres = torch.norm(barycentre - barycentre_old, p=float("inf")).item()
            elif type(lags['barycentre'])==float:
                barycentre_old = barycentre.clone()
                barycentre = (1-lags['barycentre'])*barycentre + lags['barycentre']*balanced_log_barycentre_updates(dp, d, eps, debiasing=debiasing)
                err_barycentres = torch.norm(barycentre - barycentre_old, p=float("inf")).item()

            # update the barycentre in the dictionary
            for edge in dp.graph.edges:
                dp.data_dict[edge[0]]["density"] = barycentre
            
        else:
            # need another error term to check convergence
            err_barycentres = err_potentials


        # project on second edge corresponding to the barycentre
        for edge in dp.graph.edges:
            # project on barycentre nodes edges[0]
            new_a = log_sinkhorn_update(dp, edge[0], edge, eps, rho, aprox="balanced", d=None, debiasing=True)
            if torch.any(torch.isnan(new_a)) or torch.any(torch.isinf(new_a)):
                raise ValueError("A NaN detected in sinkhorn update", new_a.sum().item(), count_iterates, edge)
            # calculate quasi convergnece
            err_potentials = max(
                err_potentials,
                torch.norm(new_a - dp.data_dict[edge[0]]["f"], p=float("inf")).item(),
            )
            dp.data_dict[edge[0]]["f"] = new_a
        
        # Update debiasing potential
        if debiasing and fixed_barycentre is None: # if fixed barycentre we don't need to update the debiasing potential as it depends on the barycentre grid which is fixed
            if count_iterates % debiasing_update_freq == 0 and type(debiasing_update_freq)==int and debiasing_update_freq > 0 :
                d = debiasing_dual_potential_update(dp, d, barycentre, eps)
                # Attach potential to the graph - all pointing to the same item
                # ToDo not have to redo this every time
                for edges in dp.graph.edges:
                    dp.data_dict[edges[0]]["debiased_potential"] = d
            elif debiasing_update_freq < 0 and type(debiasing_update_freq)==int :
                for i in range(-debiasing_update_freq):
                    d = debiasing_dual_potential_update(dp, d, barycentre, eps)
                for edges in dp.graph.edges:
                    dp.data_dict[edges[0]]["debiased_potential"] = d    
            elif type(debiasing_update_freq) == float:
                d = (1-debiasing_update_freq)*d + debiasing_update_freq*debiasing_dual_potential_update(dp, d, barycentre, eps)
                for edges in dp.graph.edges:
                    dp.data_dict[edges[0]]["debiased_potential"] = d

        # Tolerance and err_potentials or checks
        potential_error_list.append(err_potentials)
        barycentre_error_list.append(err_barycentres)
        
        # constraints on all variables
        if measure_constraints:
            sum_b, sum_f, sum_d, sum_g = measure_constraint_log_sinkhorn_update(dp, barycentre, epsilon, rho, aprox, d, debiasing)

            constraints_dict['partial_d'].append(sum_d)
            constraints_dict['partial_bary'].append(sum_b)
            constraints_dict['partial_f'].append(sum_f)
            constraints_dict['partial_g'].append(sum_g)

            constraint_err = np.mean([abs(sum_b), abs(sum_f), abs(sum_d), abs(sum_g)])
        else:
            err = err_barycentres
        count_iterates += 1


        if energy_tracking:
            _, _, breakdown_dict = asymmetric_cost(
                dp, 
                epsilon, 
                rho, 
                aprox, 
                debiasing, 
                return_breakdown=True, 
                ignore_const=True)
            
            energy_dict['total_cost'].append(breakdown_dict['total_cost'])
            energy_dict['unbalanced_sinkhorn_terms'].append(breakdown_dict['unbalanced_sinkhorn_terms'])
            energy_dict['debiasing_term'].append(breakdown_dict['debiasing_term'])
            energy_dict['uot_mu_mu_terms'].append(breakdown_dict['uot_mu_mu_terms'])
           

        # This will always reach the correct epsilon eventually
        # as tol is increased before reevaluting the while loop
        if epsilon_annealing:
            if verbose:
                print(f'Current epsilon: {eps.item():.4e}, iteration: {count_iterates}, potential error: {err_potentials:.4e}, barycentre error: {err_barycentres:.4e}')
            # converge to a lower tolerance
            if (err_barycentres < tol * 10 and count_epsilon < len(epsilon_list) - 1) or (count_iterates % 3 == 0):
                count_epsilon += 1
                eps = epsilon_list[count_epsilon].view(-1, 1)
                err_barycentres = tol + 1.0  # reset to continue
                if verbose:
                    print(
                        f"Sinkhorn reached tolerance for epsilon {eps.item():.4e}, continuing to epsilon {epsilon_list[count_epsilon].item():.4e}"
                    )
            if (
                count_epsilon == len(epsilon_list) - 1
                or count_iterates > max_iterates - 10
            ):
                # at final epsilon so turn off annealing
                # If given half the iterates switch
                epsilon_annealing = False
                eps = epsilon.view(-1, 1)
                if verbose:
                    print("Finishing annealing at count_iterates ", count_iterates)

            if termination_criterion == "barycentre":
                err = barycentre_error_list[-1]
            elif termination_criterion == "potential":
                err = potential_error_list[-1]
            elif termination_criterion == "constraint" and measure_constraints:
                err = constraint_err
            else:
                raise ValueError("Invalid termination criterion")
    if verbose:
        print(
            f"Sinkhorn finished after {count_iterates} iterations with barycentre error {err_barycentres} and potential error {err_potentials}"
        )

    if debiasing:
         for edges in dp.graph.edges:
            assert (
                dp.data_dict[edges[0]]["debiased_potential"] is d
            ), "Debiasing potential should be the same object"

    if measure_constraints and not energy_tracking:
        return data_processor, barycentre, potential_error_list, barycentre_error_list, constraints_dict
    elif energy_tracking and not measure_constraints:
        return data_processor, barycentre, potential_error_list, barycentre_error_list, energy_dict
    elif energy_tracking and measure_constraints:
        return data_processor, barycentre, potential_error_list, barycentre_error_list, constraints_dict, energy_dict
    else:
        return data_processor, barycentre, potential_error_list, barycentre_error_list


def _log_reduction_for_sinkhorn(dp, k, edge, epsilon, d=None, debiasing=True):
    """
    Log version of;

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

    # holder for reduction
    d_temp = torch.ones_like(dp.data_dict[bary_node]["f"])

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
        # temp = _tensorised_log_sinkhorn_reduction_stabalised(
            f,
            d_temp,
            ind,
            dp.data_dict[edge]["x1y1"],
            dp.data_dict[edge]["x2y2"],
            epsilon,
        )

        # testing for NaN/inf
        # if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        #     raise ValueError("Tensorised reduction NaN/inf detected", temp.sum().item(), k, edge)

    # Otherwise PyKeOps
    elif "grid" in dp.data_dict[edge[0]] and "grid" in dp.data_dict[edge[1]]:
        temp = _flat_grid_log_sinkhorn_reduction(
            f,
            d_temp,
            ind,
            dp.data_dict[k]["grid"],
            dp.data_dict[edge[1] if edge[0] == k else edge[0]]["grid"],
            epsilon,
        )

        # testing for NaN/inf
        # if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        #     raise ValueError("Flat grid reduction NaN/inf detected", temp.sum().item(), k, edge)
    
    assert temp.shape == dp.data_dict[edge[0] if k == edge[1] else edge[1]]["f"].shape, "Reduction shape incorrect"
    
    return temp + np.log(1/np.prod(f.shape))


def _bary_reduction_for_sinkhorn(dp, k, edge, epsilon):
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

    # holder for reduction
    d_temp = torch.ones_like(dp.data_dict[bary_node]["f"])

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
        
        # temp = _tensorised_log_sinkhorn_reduction(
        # temp = _tensorised_log_sinkhorn_reduction_stabalised(
        #     f,
        #     d_temp,
        #     ind,
        #     dp.data_dict[edge]["x1y1"],
        #     dp.data_dict[edge]["x2y2"],
        #     epsilon,
        # )
        temp = _tensorised_sinkhorn_reduction(
            torch.exp(f/epsilon), 
            dp.data_dict[edge]["x1y1"],
            dp.data_dict[edge]["x2y2"],
            epsilon, 
            d_temp,
            ind
            )

        # testing for NaN/inf
        # if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        #     raise ValueError("Tensorised reduction NaN/inf detected", temp.sum().item(), k, edge)

    # Otherwise PyKeOps
    elif "grid" in dp.data_dict[edge[0]] and "grid" in dp.data_dict[edge[1]]:
        temp = _flat_grid_sinkhorn_reduction(
            torch.exp(f/epsilon), 
            dp.data_dict[k]["grid"],
            dp.data_dict[edge[1] if edge[0] == k else edge[0]]["grid"], 
            epsilon, 
            d_temp,
            ind)
        
        # I can do a max shift version but then needs to take the exp of the max which is unstable
        # temp = _flat_grid_log_sinkhorn_reduction(
        #     f,
        #     d_temp,
        #     ind,
        #     dp.data_dict[k]["grid"],
        #     dp.data_dict[edge[1] if edge[0] == k else edge[0]]["grid"],
        #     epsilon,
        # )

        # testing for NaN/inf
        # if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        #     raise ValueError("Flat grid reduction NaN/inf detected", temp.sum().item(), k, edge)
    
    assert temp.shape == dp.data_dict[edge[0] if k == edge[1] else edge[1]]["f"].shape, "Reduction shape incorrect"
    
    return temp /np.prod(f.shape)

def debiasing_dual_potential_update(dp, d, barycentre, epsilon, return_s=False):
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
        # a, X, Y, epsilon, d=None, ind=None
        s = _flat_grid_sinkhorn_reduction(
            d,
            dp.data_dict[edge[0]]["grid"],
            dp.data_dict[edge[0]]["grid"],
            epsilon,
        )

    # add constants to s
    s /= np.prod(d.shape)**2

    if return_s:
        return s
    else:
        return torch.sqrt(d * barycentre / s)

    

def log_sinkhorn_update(dp, k, edge, epsilon, rho, aprox, d, debiasing, zero_tol=1e-12):
    """

    Wanted behaviour: given node k and edge (k,j) or (j,k) perform the reduction
    to the node k. To summing against j

    old behvaiour: Given index k and edge (k,j) or (j,k) perform the reduction again the index k
    meaning the output with be 'on' node j.

    """

    assert k in edge

    #  reduction across the opposite potential
    # dp, edge[1] if k == edge[0] else edge[0], edge, epsilon, d, debiasing=debiasing
    pot_before = dp.data_dict[k]["f"].sum().item()
    
    s = epsilon*_log_reduction_for_sinkhorn(
        dp, edge[1] if k == edge[0] else edge[0], edge, epsilon, d, debiasing=debiasing
    )

    # if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
    #     raise ValueError("Before sum",pot_before, s.sum().item(), k, edge, epsilon)

    # constants;
    s += epsilon* np.log(1/np.prod(dp.data_dict[k]["f"].shape)) 

    # if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
    #     raise ValueError("log_sinkhorn_update NaN/inf detected in sinkhorn update", s.sum().item(), k, edge, epsilon)

    # 'add' data terms s = eps log(data) - s
    data = dp.data_dict[k]["density"]
    if aprox == "balanced":
        temp = torch.where(data > zero_tol , epsilon * torch.log(data) - s, -1e3*torch.ones_like(s))
    elif aprox == 'kl':
        temp = torch.where(data > zero_tol , epsilon * torch.log(data) - s, -1e3*torch.ones_like(s))
        # contract
        temp *= rho/ (rho + epsilon)
    elif aprox == 'tv':
        temp = torch.where(data > zero_tol , epsilon * torch.log(data) - s, -1e3*torch.ones_like(s))
        # contract - maybe clamp  then where? 
        temp = torch.clamp(temp, min=-rho, max=rho)

    if torch.any(torch.isnan(temp)) or torch.any(torch.isinf(temp)):
        raise ValueError("log_sinkhorn_update NaN/inf detected in sinkhorn update", temp.sum().item(), k, edge, epsilon)

    return temp


def balanced_log_barycentre_updates(dp: SinkhornDataProcessor, d, epsilon, debiasing):
    """
    I'm not sure hoe to separate this fully from the
    dictionary structure, without creating the reductions outwise the loop?
    But this would require a lot of memory. So think its better to just calcalte with the dictionary
    """

    barycentre = d.clone() # becasue we've pulled d out the front
    for e1, e2, w in dp.graph.edges(data=True):
        # s = _log_reduction_for_sinkhorn(dp, e2, (e1, e2), epsilon) # d attacched before
        # s += np.log(1/np.prod(dp.data_dict[e1]["f"].shape)) # to counteract the log reduction normalisation
        # barycentre *= torch.exp(s) ** w["weight"]
        s = _bary_reduction_for_sinkhorn(dp, e2, (e1, e2), epsilon) # d attacched before
        s /= np.prod(dp.data_dict[e1]["f"].shape) # to counteract the log reduction normalisation
        barycentre *= s ** w["weight"]
    # check broadcasting is correct
    assert barycentre.shape == d.shape

    # check for nan and infs
    if torch.any(torch.isnan(barycentre)) or torch.any(torch.isinf(barycentre)):
        raise ValueError("NaN/inf detected in barycentre update", barycentre.sum().item())

    return barycentre



# --------------------------------------------------------
#   MEASURE CONSTRAINTS
# ---------------------------------------------------------

def sum_bary_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=1e-12):
    # barycentre derivative constraint:
    sum_bary = 0
    for e1, e2, w in dp.graph.edges(data=True):
        # there will be many large values from where the density is close to zero
        # what do i do with these...
        sum_bary += torch.where(
            dp.data_dict[e1]["density"] > zero_tol,  
            dp.data_dict[e1]["f"] * w["weight"], 
            torch.zeros_like(barycentre)) 
        # sum_bary += dp.data_dict[e1]["f"] * w["weight"]
    if debiasing:
        sum_bary -=  torch.where(d > zero_tol, epsilon *torch.log(d), torch.zeros_like(d))

    return sum_bary.mean().item()

def sum_f_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=1e-12):
     # f (potenital on barycentre) constraint: balanced constraint avergage over leaf nodes
    sum_f = 0
    for e1, e2, w in dp.graph.edges(data=True):
        marginal = calculate_node_marginal(dp, e1, epsilon, debiasing=False)[0]
        sum_f += (barycentre - marginal)* w["weight"]
    
    return sum_f.mean().item()

def sum_d_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=1e-12):
    # debiasing dual potential constraint: 
    if debiasing:
        s = debiasing_dual_potential_update(dp, d, barycentre, epsilon, return_s=True)
        sum_d = - barycentre / (d + zero_tol) + s
    else:
        sum_d = torch.zeros_like(barycentre)
    return sum_d.mean().item()

def sum_g_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=1e-12):
    for e1, e2, w in dp.graph.edges(data=True):
        assert e1 % 2 == 0, "e1 should be the barycentre node"
        assert e2 % 2 == 1, "e2 should be the data node"

    # g constraint: now depends on the aprox func
    sum_g = []
    if aprox == "balanced":
        for e1, e2, w in dp.graph.edges(data=True):
            marginal = calculate_node_marginal(dp, e2, epsilon, debiasing=False)[0]
            sum_g.append((torch.where(dp.data_dict[e2]["density"] > zero_tol, dp.data_dict[e2]["density"] - marginal, torch.zeros_like(marginal)) * w["weight"]).sum().item())
    if aprox == 'kl':
        for e1, e2, w in dp.graph.edges(data=True):
            marginal = calculate_node_marginal(dp, e2, epsilon, debiasing)[0]
            sum_g.append((torch.where(dp.data_dict[e2]["density"] > zero_tol, torch.exp(-dp.data_dict[e2]["f"]/rho)*dp.data_dict[e2]["density"] - marginal, torch.zeros_like(marginal)) * w["weight"]).sum().item())
    if aprox == 'tv':
        for e1, e2, w in dp.graph.edges(data=True):

            assert  (dp.data_dict[e2]["f"] <= rho).all(), "f should be clamped to +- rho in TV aproximation"
            marginal = calculate_node_marginal(dp, e2, epsilon, debiasing)[0]
            sum_g.append((torch.where(dp.data_dict[e2]["density"] > zero_tol, dp.data_dict[e2]["density"] - marginal, torch.zeros_like(marginal)) * w["weight"]).sum().item())

    return np.mean(sum_g)

def measure_constraint_log_sinkhorn_update(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=1e-12):

    sum_bary = sum_bary_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=zero_tol)

    sum_f = sum_f_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=zero_tol)
    
    sum_d = sum_d_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=zero_tol)

    sum_g = sum_g_measure_constraint(dp, barycentre, epsilon, rho, aprox, d, debiasing, zero_tol=zero_tol)
    
    return sum_bary, sum_f, sum_d, sum_g
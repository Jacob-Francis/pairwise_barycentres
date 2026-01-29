import torch
import numpy as np
from graph_dp import SinkhornDataProcessor
from .utils import _flat_grid_sinkhorn_reduction, chizat_proxdiv_step, tensorise_f, generate_epsilon_list, process_dict_for_barycentre, _tensorised_sinkhorn_reduction
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
    debiasing_update_freq: int = 1,
    mass_scaling=False,
    fixed_barycentre=None,
    bary_animation=False
):
    # shorten to pass around
    dp = data_processor

    process_dict_for_barycentre(dp, debiasing=debiasing)

    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)

    # Initalise the deibasing potential with barycentre shape
    d = dp._torch_numpy_process(torch.ones_like(dp.data_dict[0]["density"]))

    if fixed_barycentre is None:
        barycentre = d.clone() / d.sum()
        barycentre_old = d.clone() / d.sum()
    else:
        barycentre = dp._torch_numpy_process(fixed_barycentre).reshape(*d.shape)
        barycentre_old = dp._torch_numpy_process(fixed_barycentre).reshape(*d.shape)
        # update the barycentre in the dictionary
        for edge in dp.graph.edges:
            dp.data_dict[edge[0]]["density"] = barycentre

    # mass sum term
    mass_term = 0.0
    for edge in dp.graph.edges:
        mass_term += dp.graph.edges[edge]["weight"] * dp.data_dict[edge[1]]["density"].sum().item()
    
    # scale the barycentre up and densities 
    if mass_scaling and fixed_barycentre is None:
        print('Initial mass term', mass_term)
        for edge in dp.graph.edges:
            dp.data_dict[edge[0]]["density"] *= mass_term
        barycentre *= mass_term
        barycentre_old *= mass_term
    
    if epsilon_annealing:
        epsilon_list = generate_epsilon_list(epsilon)
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
    if bary_animation:
        barycentre_list = [barycentre.clone().cpu().numpy()]

    while count_iterates < max_iterates and err_barycentres > tol:
        # print('current error', potential_error_list[-1] if len(potential_error_list)>0 else 'N/A', 'barycentre error', barycentre_error_list[-1] if len(barycentre_error_list)>0 else 'N/A')
        # if debiasing_update_freq > 0 and count_iterates % debiasing_update_freq == 0:
        #     # reset errors
        err_potentials = -np.inf
        err_barycentres = -np.inf

        # Project edge corresponding to the data
        # I could stick these in paralell on the gpu - but for 200 by 200 I'm had problems with memory

        for edge in dp.graph.edges:
            # project on barycentre nodes edges[1]
            new_b = sinkhorn_update(dp, edge[1], edge, eps, rho, aprox, d, debiasing=debiasing)
            # new_b = sinkhorn_update(dp, edge[1], edge, eps, rho, aprox, torch.ones_like(d), debiasing=False)

            # calculate quasi convergnece

            err_potentials = max(
                err_potentials,
                torch.norm(torch.where(new_b > 0, torch.log(new_b) - torch.log(dp.data_dict[edge[1]]["a"]), torch.zeros_like(new_b)), p=float("inf")).item(),
            )
            dp.data_dict[edge[1]]["a"] = new_b

        # Barycentre updates and update barycentre in dictionary
        # if debiasing_update_freq > 0 and count_iterates % debiasing_update_freq == 0:
        if fixed_barycentre is None:
            barycentre_old = barycentre.clone()

            barycentre = balanced_barycentre_updates(dp, d, eps)

            # mass scaling
            monitor_mass_stabilty = barycentre.sum().item()
            if mass_scaling:
                for _ in  range(3):
                    barycentre = torch.exp(mass_term - barycentre.sum()) * barycentre
                    # print('change in mass', barycentre.sum().item() - monitor_mass_stabilty, mass_term, barycentre.sum().item())
                    monitor_mass_stabilty = barycentre.sum().item()


            # calcualte error to old barycentre
            err_barycentres = torch.norm(barycentre - barycentre_old, p=float("inf")).item()

            # update the barycentre in the dictionary
            for edge in dp.graph.edges:
                dp.data_dict[edge[0]]["density"] = barycentre
            
            if bary_animation:
                barycentre_list.append(barycentre.clone().cpu().numpy())
        else:
            # need another error term to check convergence
            err_barycentres = err_potentials

        # project on second edge corresponding to the barycentre
        for edge in dp.graph.edges:
            # project on barycentre nodes edges[0]
            new_a = sinkhorn_update(dp, edge[0], edge, eps, rho, aprox="balanced", d=d, debiasing=debiasing)
            # new_a = sinkhorn_update(dp, edge[0], edge, eps, rho, aprox="balanced", d=torch.ones_like(d), debiasing=False)

            # calculate quasi convergnece
            err_potentials = max(
                err_potentials,
                torch.norm(torch.where(new_a > 0, torch.log(new_a) - torch.log(dp.data_dict[edge[0]]["a"]), torch.zeros_like(new_a)), p=float("inf")).item(),
            )
            dp.data_dict[edge[0]]["a"] = new_a # multiply by debiasing potential                

        # Update debiasing potential
        if debiasing and fixed_barycentre is None:
            if debiasing_update_freq > 0 and count_iterates % debiasing_update_freq == 0:
                d = debiasing_dual_potential_update(dp, d, barycentre, eps)
            elif debiasing_update_freq == 0:
                raise Warning("Debiasing update freq cannot be zero")
            elif debiasing_update_freq < 0:
                for i in range(-debiasing_update_freq):
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
                        f"Sinkhorn reached tolerance continuing to epsilon {epsilon_list[count_epsilon].item():.4e}"
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

    if bary_animation:
        return data_processor, barycentre, potential_error_list, barycentre_error_list, barycentre_list

    return data_processor, barycentre, potential_error_list, barycentre_error_list

def _chizat_reduction_for_sinkhorn(dp, k, edge, epsilon, d=None, debiasing=False):
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
        if d is None:
            d = torch.ones_like(dp.data_dict[bary_node]["a"])
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


def Kd_dual_potential_reduction(dp, d, epsilon):
    """
    Kd reduction for debiasing potential update
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
    
    return s

def debiasing_dual_potential_update(dp, d, barycentre, epsilon):
    """
    Debiasing requires that we know th grid for the barycentre and this may be tensorisable
    in which case we need an x1x1, x2x2 type thing in the dictionary. If the grids are the same
    then x1y1==x1x1.

    SSSSooo
    """

    shape_temp = d.shape

    s = Kd_dual_potential_reduction(dp, d, epsilon)
    
    # checking output
    output = torch.sqrt(d * barycentre / s)
    if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
        raise ValueError("Debiasing potential update NaN/inf detected", output.sum().item())
    if torch.any(output < 0):
        raise ValueError("Debiasing potential update negative or zero values detected", output.min().item())

    assert output.shape == shape_temp

    return output

def sinkhorn_update(dp, k, edge, epsilon, rho, aprox, d=None, debiasing: bool = False):
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




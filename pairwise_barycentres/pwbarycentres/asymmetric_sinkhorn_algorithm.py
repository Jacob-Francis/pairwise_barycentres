import torch
import numpy as np
from data_processing import SinkhornDataProcessor
from pykeops_formulas import chizat_marginals, chizat_reduction
from utils import chizat_proxdiv_step, tensorise_f

def generate_epsilon_list(epsilon: float, max_iterates: int) -> list:
    epsilon_list = torch.logspace(
        torch.log2(0.5),          # log2 of the start
        torch.log2(epsilon.item()),  # log2 of the end
        steps=10,
        base=2
        )
    
    return epsilon_list.to(epsilon)

def asymmetric_sinkhorn_algorithm(
    data_processor: SinkhornDataProcessor,
    epsilon: float,
    rho: float,
    aprox: str,
    max_iterates: int,
    tol: float,
    epsilon_annealing: bool = False,
    debiasing: bool = True,
):
    # shorten to pass around
    dp = data_processor

    process_dict_for_barycentre(dp, debiasing=debiasing)
    
    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)

    # Initalise the deibasing potential with barycentre shape
    d = dp._torch_numpy_process(torch.ones_like(dp.data_dict[0]['density']))
    barycentre = d.clone()
    barycentre_old = d.clone()

    if epsilon_annealing:
        epsilon_list = generate_epsilon_list(epsilon, max_iterates)
        count_epsilon = 0
        eps = epsilon_list[count_epsilon]
    else:
        eps = epsilon
        count_epsilon = None

    # Initialise parameters and lists
    count_iterates = 0
    err_potentials = tol + 1.0
    err_barycentres = tol + 1.0
    potential_error_list = []
    barycentre_error_list = []

    while count_iterates < max_iterates and err_barycentres > tol:
        # Project edge corresponding to the data
        # I could stick these in paralell on the gpu - but for 200 by 200 I'm had problems with memory

        for edge in dp.graph.edges:
            # project on barycentre nodes edges[1]
            new_b = sinkhorn_update(dp, edge[1], edge, eps, rho, aprox)
            # calculate quasi convergnece
            err_potentials = max(err_potentials, torch.norm(new_b - dp.data_dict[edge[1]]['a'], p=float('inf')).item())
            dp.data_dict[edge[1]]['a'] = new_b
            

        # Barycentre updates and update barycentre in dictionary
        barycentre_old = barycentre.clone()
        barycentre = balanced_barycentre_updates(dp, d, eps)
        # calcualte error to old barycentre
        err_barycentres = torch.norm(barycentre - barycentre_old, p=float('inf')).item()
        # update the barycentre in the dictionary
        for edge in dp.graph.edges:
            dp.data_dict[edge[0]]['density'] = barycentre

        # project on second edge corresponding to the barycentre
        for edge in dp.graph.edges:
            # project on barycentre nodes edges[0]
            new_a = sinkhorn_update(dp, edge[0], edge, eps, rho, aprox='balanced')
            # calculate quasi convergnece
            err_potentials = max(err_potentials, torch.norm(new_a - dp.data_dict[edge[0]]['a'], p=float('inf')).item())
            dp.data_dict[edge[0]]['a'] = new_a

        # Update debiasing potential
        if debiasing:
            d = debiasing_dual_potential_update(dp, d, barycentre, eps)
        
        
        # Tolerance and err_potentials or checks
        potential_error_list.append(err_potentials)
        barycentre_error_list.append(err_barycentres)

    return data_processor, barycentre, potential_error_list, barycentre_error_list

def _flat_grid_sinkhorn_reduction(a,X,Y,epsilon):

    # kernel computations - K @ a
    # main bottle neck
    return chizat_reduction(
        X,
        Y,
        epsilon,
        a
    )


def _tensorised_sinkhorn_reduction(a, x1y1, x2y2,epsilon):

    # kernel computations - K @ a
    # main bottle neck
    return tensorise_f(
        torch.exp(-x1y1/epsilon),
        torch.exp(-x2y2/epsilon),
        a
    )

def _chizat_reduction_for_sinkhorn(dp, k, edge, epsilon):
    assert k in edge

    # Can I tensorise?
    if 'x1y1' in dp.data_dict[edge] and 'x2y2' in dp.data_dict[edge]:
        return _tensorised_sinkhorn_reduction(
            dp.data_dict[k]['a'],
            dp.data_dict[edge]['x1y1'],
            dp.data_dict[edge]['x2y2'],
            epsilon,
        )
    # Otherwise PyKeOps
    elif 'grid' in dp.data_dict[edge[0]] and 'grid' in dp.data_dict[edge[1]]:
        return _flat_grid_sinkhorn_reduction(
            dp.data_dict[k]['a'],
            dp.data_dict[k]['grid'],
            dp.data_dict[edge[1] if edge[0]==k else edge[0]]['grid'],
            epsilon,
        )

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
    if 'x1x1' in dp.data_dict[edge[0]] and 'x2x2' in dp.data_dict[edge[0]]:
        s = _tensorised_sinkhorn_reduction(
            d,
            dp.data_dict[edge[0]]['x1x1'],
            dp.data_dict[edge[0]]['x2x2'],
            epsilon,
        )
    # Otherwise PyKeOps
    elif 'grid' in dp.data_dict[edge[0]]:
        s = _flat_grid_sinkhorn_reduction(
            d,
            dp.data_dict[edge[0]]['grid'],
            dp.data_dict[edge[0]]['grid'],
            epsilon,
        )
    
    return torch.sqrt(
        d * barycentre / s
    )

def sinkhorn_update(dp, k, edge, epsilon, rho, aprox):
    """
    
    Given index k and edge (k,j) or (j,k) perform the reduction again the index k
    meaning the output with be 'on' node j.

    """
    
    assert k in edge

    s = _chizat_reduction_for_sinkhorn(dp, k, edge, epsilon)

    return chizat_proxdiv_step(
        s,
        epsilon,
        rho,
        dp.data_dict[edge[1] if edge[0]==k else edge[0]]['density'],
        aprox=aprox,
        )
    
def balanced_barycentre_updates(dp: SinkhornDataProcessor, d, epsilon):
    """
    I'm not sure hoe to separate this fully from the
    dictionary structure, without creating the reductions outwise the loop?
    But this would require a lot of memory. So think its better to just calcalte with the dictionary
    """

    barycentre = d.clone()
    for edge, w in dp.graph.edges(data=True):
        s = _chizat_reduction_for_sinkhorn(dp, edge[0], edge, epsilon)
        barycentre *= s**w['weight']

    # check broadcasting is correct
    assert barycentre.shape == d.shape  

    return barycentre
    
def process_dict_for_barycentre(dp: SinkhornDataProcessor, debiasing=True):
    """
    Ensure that the barycentre nodes have the same density and a potential
    """

    # ToDo: check dp set up correctly
    # I suppose we could actually slove the problem thorugh different grids in which case they'd 
    # have different grids but lets leave that for now
    for edge1 in dp.graph.edges:
        for edge2 in dp.graph.edges:
            assert dp.data_dict[edge1[0]]['density'] is dp.data_dict[edge2[0]]['density'], "Barycentre node should have the same data"
    
    if debiasing:
        edge2 = list(dp.graph.edges)[0]

        # Need to add x1x1 x2x2 for debiasing potential term
        if 'x1y1' in dp.data_dict[edge1] and 'x2y2' in dp.data_dict[edge1]:
            # we can tensorise, so we must be able to tensorize the symmetric problem

            # Does eveyone have the same grid?
           
            assert edge1 != edge2
            if dp.data_dict[edge1]['x1y1'] is dp.data_dict[edge2]['x1y1']:
                # then the edges are sharing a grid
                # So assign to the firs edge barycentre node the x1x1, x2x2
                dp.data_dict[edge2]['x1x1'] = dp.data_dict[edge2]['x1y1']
                dp.data_dict[edge2]['x2x2'] = dp.data_dict[edge2]['x2y2']
            else:
                # Not eveyone shares the same grid so need to compute symmetric version
                # it should still have a grid associated
                grid = dp.data_dict[edge2[0]]['grid']
                if isinstance(grid, tuple) :
                    dp.data_dict[edge2[0]]["x1x1"], dp.data_dict[edge2[0]]["x1x1"] = dp._cost_for_tuple(grid, grid)

                elif len(grid.shape) == 3 and len(grid.shape) == 3:
                    n1, n2, n3 = grid.shape
                    assert n3 == 2 , "We assume 2D points"

                    # Calculate cost matrices - the indexing works
                    # because torch cdist eliminats the common axis which will have the same values.
                    dp.data_dict[edge2[0]]["x1y1"], dp.data_dict[edge2[0]]["x2y2"] = dp._cost_for_meshgrid(grid, grid, n1, n2, n1, n2)
                    

        elif 'grid' in dp.data_dict[edge2[0]]:
            pass # we can use PyKeOps
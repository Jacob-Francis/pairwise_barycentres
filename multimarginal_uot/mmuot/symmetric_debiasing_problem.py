"""
If we are given a central node and then all leaves are the same node, then we need only one kernel computation
to compute all the costs from the center to the leaves. i.e. it becomes similar to the two marginal problem,
but with a scaling by the number of members. This can also then be used to compute the problem with all the same node data,.

"""
import numpy as np
import torch
import pwbarycentres as pwb
from graph_dp import SinkhornDataProcessor
from .mmuot_sinkhorn_graph_reductions import graph_creator_from_edges_and_weights

def muot_symmetric_problem(centre, leaf, grid1, grid2, M, epsilon, rho, aprox, max_iterates=2000, tol=1e-9, cuda_device=None):
    """
    Solve the symmetric debiasing problem for multimarginal UOT where all leaves are the same node data.

    Args:
        centre: density of the central node (barycentre)
        leaf: density of the leaf nodes (all the same)
        grid1: grid for the central node
        grid2: grid for the leaf nodes
        M: number of leaf nodes
        epsilon: entropic regularisation parameter
        rho: unbalanced regularisation parameter
        aprox: type of approximation ('kl', 'l2', etc.)
        max_iterates: maximum number of Sinkhorn iterations
        tol: tolerance for convergence
    """

    # process data - this is really a dump class for processing inputs
    edges = [(0,1)]
    counter = 0
    data_dict = {
        0: {
            "density": centre,
            "grid": grid1,
        },
        1: {
            "density": leaf,
            "grid": grid2,
        },
    }

    graph = graph_creator_from_edges_and_weights(edges, weights=None)

    # build data processor
    dp = SinkhornDataProcessor(
        potentials="f",
        data_dict=data_dict,
        graph=graph,
        free_grids=False,
        cuda_device=cuda_device,
    )

    # process data and espilons etc
    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)
    centre = dp._torch_numpy_process(centre)
    leaf = dp._torch_numpy_process(leaf)

    lse = generate_lse_function(dp, grid1, grid2, epsilon, M=M)

    # initialise potentials
    f = torch.zeros_like(centre)
    g = torch.zeros_like(leaf)

    count = 0
    err = torch.inf

    while count < max_iterates and err > tol:
        f_prev = f.clone()
        g_prev = g.clone()

        # Update f
        f = epsilon*lse(g, ind=1)
        f = -pwb.utils.log_aprox_step(
            f*M, # scale by M number of leaves
            epsilon,
            rho,
            centre, 
            aprox=aprox,
        )

        # Update g
        g = epsilon*lse(f, ind=0)
        g = -pwb.utils.log_aprox_step(
            g, 
            epsilon,
            rho,
            leaf*M,
            aprox=aprox,
        )

        # damp updates? 
        if count > 0:
            f = 0.5 * (f + f_prev)
            g = 0.5 * (g + g_prev)
       
        # Compute convergence error
        err_f = torch.linalg.norm(f - f_prev, ord=float('inf'))
        err_g = torch.linalg.norm(g - g_prev, ord=float('inf'))
        err = max(err_f, err_g)
        print(f"errs", err_f.item(), err_g.item())
        count += 1

    print(f"Converged in {count} iterations with error {err.item()}")

    return f, g, dp


def generate_lse_function(dp, grid1, grid2, epsilon, M=1):
    # We first need to figure out if we can tensoirse or not and then create the lse function
    # thankfully neither include the M sclaing yet.
    tensoirse = False


    if isinstance(grid1, tuple) and isinstance(grid2, tuple):
        xx, yy = dp._cost_for_tuple(grid1, grid2)
        tensoirse = True
        
    elif len(grid1.shape) == 3 and len(grid2.shape) == 3:
        n1, n2, n3 = grid1.shape
        m1, m2, m3 = grid2.shape

        xx, yy = (
                dp._cost_for_meshgrid(grid1, grid2, n1, n2, m1, m2)
            )
        tensoirse = True
    
    # else we use pykeops

    if tensoirse:
        # tensorise f actually figures out which dimension - though based on shape - but oh well
        lse = lambda f, ind : _tensorised_log_sinkhorn_reduction(
                f,
                ind,
                np.sqrt(M) *xx,
                np.sqrt(M) *yy,
                epsilon,
            )

    else:
        # pykeops
        lse = lambda f, ind : _flat_grid_log_sinkhorn_reduction(
            f,
            ind,
            np.sqrt(M) *grid1,
            np.sqrt(M) *grid2,
            epsilon,
        )
        
    return lse

def _tensorised_log_sinkhorn_reduction(f, ind, x1y1, x2y2, epsilon):
    """
    f dual potential being reduced over
    d debiasing potential
    ind - 0 if d and f go together (and are summed), 1 if they are opposite so d is not summed. 
    """
    # kernel computations - K @ a
    # main bottle neck
    if ind==0:
        temp = pwb.utils.tensorise_f(torch.exp((-x1y1) / epsilon), torch.exp((-x2y2) / epsilon), torch.exp(f / epsilon))
    else:
        temp = pwb.utils.tensorise_f(torch.exp((-x2y2) / epsilon), torch.exp((-x1y1) / epsilon), torch.exp(f / epsilon))

    return torch.log(temp)

def _flat_grid_log_sinkhorn_reduction(f, ind, X, Y, epsilon):
    """
    if f and d are both Vi then ind= 0
    if f is Vi and d is Vj then ind=1
    """
    #something in numpy still

    # kernel computations - K @ a
    # main bottle neck
    if ind==0:
        return  pwb.pykeops_formulas.log_reduction_ii(f, X, Y, epsilon, d=None)
    elif ind==1:
        return pwb.pykeops_formulas.log_reduction_ii(f, Y, X, epsilon, d=None)
    

def symmetric_cost(centre, leaf, grid1, grid2, no_members, epsilon, rho, aprox, max_iterates=2000, tol=1e-9, cuda_device=None):

    f, g, dp = muot_symmetric_problem(centre, leaf, grid1, grid2, no_members, epsilon, rho, aprox, max_iterates=2000, tol=1e-9, cuda_device=cuda_device)

    # process data 
    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)
    centre = dp._torch_numpy_process(centre)
    leaf = dp._torch_numpy_process(leaf)

    term1 = no_members*pwb.utils._dual_cost_data_term_f_potential(g, leaf, aprox, epsilon, rho)
    term2 = pwb.utils._dual_cost_data_term_f_potential(f, centre, aprox, epsilon, rho)

    if isinstance(grid1, tuple) and isinstance(grid2, tuple):
        gridding = dp._cost_for_tuple(grid1, grid2)
        tensoirse = True
        
    elif len(grid1.shape) == 3 and len(grid2.shape) == 3:
        n1, n2, n3 = grid1.shape
        m1, m2, m3 = grid2.shape

        gridding = (
                dp._cost_for_meshgrid(grid1, grid2, n1, n2, m1, m2)
            )
        tensoirse = True
    
    # else we use pykeops

    if tensoirse:
        pi_sum = (pwb.marginals._tensorised_marginal_reduction(
            *gridding,
            epsilon,
            torch.exp(f/epsilon/no_members),
            torch.exp(g/epsilon),
        ).sum())**no_members

        cost_const = (pwb.marginals._tensorised_marginal_reduction(
            *gridding,
            epsilon,
            torch.ones_like(f),
            torch.ones_like(g),
        ).sum())**no_members
    else:
        # pykeops
        pi_sum = (pwb.marginals._flat_grid_marginal_reduction(
            grid, grid,
            epsilon,
            torch.exp(f/epsilon/no_members),
            torch.exp(g/epsilon),
        ).sum())**no_members
        cost_const = (pwb.marginals._flat_grid_marginal_reduction(
            grid, grid,
            epsilon,
            torch.ones_like(f),
            torch.ones_like(g),
        ).sum())**no_members

    print('terms:', term1.item(), term2.item(), pi_sum.item(), cost_const.item())
    return term1 + term2  - epsilon*(pi_sum - cost_const)

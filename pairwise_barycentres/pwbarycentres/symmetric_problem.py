

# solve UOT(mu,mu| L otimes L)
# with a single dual potential with updates
import torch
import numpy as np
from graph_dp import SinkhornDataProcessor
from .utils import _flat_grid_sinkhorn_reduction, chizat_proxdiv_step, tensorise_f, generate_epsilon_list, process_dict_for_barycentre, _tensorised_sinkhorn_reduction
from .marginals import (
    calculate_node_marginal,
    _tensorised_marginal_reduction,
    _flat_grid_marginal_reduction,
)
from .utils import _dual_cost_data_term

def symmetric_cost(dp, k, edge, epsilon, rho, aprox, max_iterates=2000, tol=1e-9):

    sym_pot = symmetric_algorithm(dp, k, edge, epsilon, rho, aprox, max_iterates, tol)

    # term1 (2 because its the same potential)
    term1 = 2*_dual_cost_data_term(sym_pot, dp.data_dict[k]['density'], aprox, epsilon, rho)

    # term2 - entropic term
    assert 'grid' in dp.data_dict[k], "You may have cleared grids incorrectly"
    grid = dp.data_dict[k]['grid']
    tensoirse = False

    # can we tensorise
    if isinstance(grid, tuple):
        gridding = dp._cost_for_tuple(grid, grid)
        tensoirse = True
        
    elif len(grid.shape) == 3:
        n1, n2, n3 = grid.shape

        gridding = (
                dp._cost_for_meshgrid(grid, grid, n1, n2, n1, n2)
            )
        tensoirse = True

    if tensoirse:
        pi_sum = _tensorised_marginal_reduction(
            *gridding,
            epsilon,
            sym_pot,
            sym_pot,
        ).sum()

        cost_const = _tensorised_marginal_reduction(
            *gridding,
            epsilon,
            torch.ones_like(sym_pot),
            torch.ones_like(sym_pot),
        ).sum()
    else:
        # pykeops
        pi_sum = _flat_grid_marginal_reduction(
            grid, grid,
            epsilon,
            sym_pot,
            sym_pot,
        ).sum()
        cost_const = _flat_grid_marginal_reduction(
            grid, grid,
            epsilon,
            torch.ones_like(sym_pot),
            torch.ones_like(sym_pot),
        ).sum()

    return term1  - epsilon*(pi_sum - cost_const)


def symmetric_algorithm(dp, k, edge, epsilon, rho, aprox, max_iterates=2000, tol=1e-9):
    update_method = symmetric_mat_vec_chizat_method(dp, k, edge, epsilon, rho, aprox)

    # I'm not sure which is better to do it in, it shoudl converg fast anyway
    sym_pot_0 = dp.data_dict[k]['a'] if 'a' in dp.data_dict[k] else torch.exp(dp.data_dict[k]['f']/epsilon)
    sym_pot, err = symmetric_sinkhorn(sym_pot_0, update_method, max_iterates, tol)

    if err > tol:
        print(f'Symmertic updates for node {k} at err {err.item()} above tol={tol}')
    
    return sym_pot


def symmetric_mat_vec_chizat_method(dp, k, edge, epsilon, rho, aprox):
    '''
    k is the node which contains data mu_k, and then solving UOT^{phi, phi}(mu_k, mu_kl)
    '''
    assert k in edge

    assert 'grid' in dp.data_dict[k], "You may have cleared grids incorrectly"
    grid = dp.data_dict[k]['grid']

    tensoirse = False

    # can we tensorise
    if isinstance(grid, tuple):
        gridding = dp._cost_for_tuple(grid, grid)
        tensoirse = True
        
    elif len(grid.shape) == 3:
        n1, n2, n3 = grid.shape

        gridding = (
                dp._cost_for_meshgrid(grid, grid, n1, n2, n1, n2)
            )
        tensoirse = True

    if tensoirse:
        lse = lambda a_g: _tensorised_sinkhorn_reduction(
                a_g,
                *gridding,
                epsilon
            )
    else:
        # pykeops
        lse = lambda a_g: _flat_grid_sinkhorn_reduction(
            a_g,
            grid, 
            grid,
            epsilon
        )

    # aprox
    def one_chizat_update(sym_pot):
        return chizat_proxdiv_step(
        lse(sym_pot),
        epsilon,
        rho,
        dp.data_dict[k]["density"],
        aprox=aprox,
    )
    
    return one_chizat_update


def symmetric_sinkhorn(sym_pot_0, update_method, max_iterates=2000, tol=1e-9):

    for _ in range(max_iterates):
        sym_pot_1 = update_method(sym_pot_0)
        
        # damped update
        sym_pot_1 = 0.5*(sym_pot_1 + sym_pot_0)

        # update size
        err = torch.linalg.norm(sym_pot_1-sym_pot_0, ord=float('inf'))

        if err < tol:
            return sym_pot_1, err
        else:
            sym_pot_0 = sym_pot_1

    return sym_pot_1, err



# solve UOT(mu,mu| L otimes L)
# with a single dual potential with updates
import torch
import numpy as np
from graph_dp import SinkhornDataProcessor
from .utils import (
    _flat_grid_sinkhorn_reduction, 
    chizat_proxdiv_step,   
    _tensorised_sinkhorn_reduction,
    _dual_cost_data_term, 
    _tensorised_log_sinkhorn_reduction, 
    _dual_cost_data_term_f_potential,
    _flat_grid_log_sinkhorn_reduction
    )

from .marginals import (
    _tensorised_marginal_reduction,
    _flat_grid_marginal_reduction,
)



def symmetric_cost(dp, k, epsilon, rho, aprox, max_iterates=2000, tol=1e-9):

    sym_pot = symmetric_algorithm(dp, k, epsilon, rho, aprox, max_iterates, tol)

    # term1 (2 because its the same potential)
    term1 = 2*_dual_cost_data_term_f_potential(sym_pot, dp.data_dict[k]['density'], aprox, epsilon, rho, dp.data_dict[k]['cell_areas'])

    # need to change below 

    # term2 - entropic term
    assert 'grid' in dp.data_dict[k], "You may have cleared grids incorrectly"
    grid = dp.data_dict[k]['grid']
    tensoirse = False
    cell_areas = dp.data_dict[k]['cell_areas']

    if cell_areas.ndim > 0:
        assert cell_areas.shape == sym_pot.shape, "Cell areas should be the same shape as the potential for the symmetric problem"
    
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
            torch.exp(sym_pot/epsilon)*cell_areas,
            torch.exp(sym_pot/epsilon)*cell_areas,
        ).sum()
 
    else:
        # pykeops
        pi_sum = _flat_grid_marginal_reduction(
            grid, 
            grid,
            epsilon,
            torch.exp(sym_pot/epsilon)*cell_areas,
            torch.exp(sym_pot/epsilon)*cell_areas,
        ).sum()

    if cell_areas.ndim == 0:
        cost_const = np.prod(sym_pot.shape)**2*cell_areas.item()**2
    else:
        cost_const = cell_areas.sum()**2

    return term1  - epsilon*(pi_sum - cost_const)


def symmetric_algorithm(dp, k, epsilon, rho, aprox, max_iterates=2000, tol=1e-9):
    epsilon = dp._torch_numpy_process(epsilon)
    update_method = symmetric_mat_f_potential_method(dp, k, epsilon, rho, aprox)

    # I'm not sure which is better to do it in, it shoudl converg fast anyway
    sym_pot_0 = epsilon*torch.log(dp.data_dict[k]['a']) if 'a' in dp.data_dict[k] else dp.data_dict[k]['f']
    sym_pot, err = symmetric_sinkhorn(sym_pot_0, update_method, max_iterates, tol)

    if err > tol:
        print(f'Symmertic updates for node {k} at err {err.item()} above tol={tol}')
    
    return sym_pot


# def symmetric_mat_vec_chizat_method(dp, k, epsilon, rho, aprox):
#     '''
#     k is the node which contains data mu_k, and then solving UOT^{phi, phi}(mu_k, mu_kl)
#     '''
#     assert 'grid' in dp.data_dict[k], "You may have cleared grids incorrectly"
#     grid = dp.data_dict[k]['grid']

#     tensoirse = False

#     # can we tensorise
#     if isinstance(grid, tuple):
#         gridding = dp._cost_for_tuple(grid, grid)
#         tensoirse = True
        
#     elif len(grid.shape) == 3:
#         n1, n2, n3 = grid.shape

#         gridding = (
#                 dp._cost_for_meshgrid(grid, grid, n1, n2, n1, n2)
#             )
#         tensoirse = True

#     if tensoirse:
#         lse = lambda a_g: _tensorised_sinkhorn_reduction(
#                 a_g,
#                 *gridding,
#                 epsilon
#             )
#     else:
#         # pykeops
#         lse = lambda a_g: _flat_grid_sinkhorn_reduction(
#             a_g,
#             grid, 
#             grid,
#             epsilon
#         )

#     # aprox
#     def one_chizat_update(sym_pot):
#         return chizat_proxdiv_step(
#         lse(sym_pot),
#         epsilon,
#         rho,
#         dp.data_dict[k]["density"],
#         aprox=aprox,
#     ) / (np.prod(dp.data_dict[k]["density"].shape)**2)
    
#     return one_chizat_update

def symmetric_mat_f_potential_method(dp, k, epsilon, rho, aprox, zero_tol=1e-40):
    '''
    k is the node which contains data mu_k, and then solving UOT^{phi, phi}(mu_k, mu_kl)
    '''
    assert 'grid' in dp.data_dict[k], "You may have cleared grids incorrectly"
    grid = dp.data_dict[k]['grid']
    assert torch.is_tensor(dp.data_dict[k]['cell_areas'])
    if dp.data_dict[k]['cell_areas'].ndim == 0:
        leb = dp.data_dict[k]['cell_areas']*torch.ones_like(dp.data_dict[k]["f"])
    else:
        leb = dp.data_dict[k]['cell_areas'] 

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
        lse = lambda a_g : _tensorised_log_sinkhorn_reduction(
                a_g,
                leb,
                0,
                *gridding,
                epsilon
        )
 
    else:
        lse = lambda a_g : _flat_grid_log_sinkhorn_reduction(
            a_g,
            leb,
            0,
            grid,
            grid, 
            epsilon
            )
      
    # aprox
    def one_chizat_update(sym_pot):
        data = dp.data_dict[k]["density"]
        s = epsilon*lse(sym_pot)        

        data = torch.clamp(data, min=zero_tol) # to avoid log of zero
        temp = epsilon * torch.log(data) - s

        if aprox == "balanced":
            pass
        elif aprox == 'kl':
            # contract
            temp *= rho/ (rho + epsilon)
        elif aprox == 'tv':
            # contract  
            temp = torch.clamp(temp, min=-rho, max=rho)

        return temp
    
    return one_chizat_update

def symmetric_sinkhorn(sym_pot_0, update_method, max_iterates=2000, tol=1e-9):

    for _ in range(max_iterates):
        # print('0', sym_pot_0[:5])

        sym_pot_1 = update_method(sym_pot_0)

        # print('1', sym_pot_1[:5])

        # damped update
        sym_pot_1 = 0.5*(sym_pot_1 + sym_pot_0)

        # print('2', sym_pot_1[:5])

        # update size
        err = torch.linalg.norm(sym_pot_1-sym_pot_0, ord=float('inf'))

        if err < tol:
            return sym_pot_1, err
        else:
            sym_pot_0 = sym_pot_1

    return sym_pot_1, err

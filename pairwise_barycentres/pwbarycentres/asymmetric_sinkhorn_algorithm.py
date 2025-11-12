import torch
import numpy as np
from .data_processing import SinkhornDataProcessor
from .pykeops_formulas import chizat_marginals, chizat_reduction
from .utils import chizat_proxdiv_step, tensorise_f, _dual_cost_data_term

def generate_epsilon_list(epsilon: float, max_iterates: int) -> list:
    epsilon_list = torch.logspace(
        torch.log2(torch.tensor(0.5)),          # log2 of the start
        torch.log2(epsilon.squeeze()),  # log2 of the end
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
    verbose: bool = False,
):
    # shorten to pass around
    dp = data_processor

    process_dict_for_barycentre(dp, debiasing=debiasing)
    
    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)

    # Initalise the deibasing potential with barycentre shape
    d = dp._torch_numpy_process(torch.ones_like(dp.data_dict[0]['density']))
    barycentre = d.clone() / d.sum()
    barycentre_old = d.clone() / d.sum()

    if epsilon_annealing:
        epsilon_list = generate_epsilon_list(epsilon, max_iterates)
        count_epsilon = 0
        eps = epsilon_list[count_epsilon].view(-1, 1)
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
            new_b = sinkhorn_update(dp, edge[0], edge, eps, rho, aprox)
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
            new_a = sinkhorn_update(dp, edge[1], edge, eps, rho, aprox='balanced')
            # calculate quasi convergnece
            err_potentials = max(err_potentials, torch.norm(new_a - dp.data_dict[edge[0]]['a'], p=float('inf')).item())
            dp.data_dict[edge[0]]['a'] = new_a

        # Update debiasing potential
        if debiasing:
            d = debiasing_dual_potential_update(dp, d, barycentre, eps)
        
        # Tolerance and err_potentials or checks
        potential_error_list.append(err_potentials)
        barycentre_error_list.append(err_barycentres)

        count_iterates += 1

        if epsilon_annealing:
            if err_barycentres < tol:
                count_epsilon += 1
                eps = epsilon_list[count_epsilon].view(-1, 1)
                err_barycentres = tol + 1.0  # reset to continue

    if verbose:
        print(f'Sinkhorn finished after {count_iterates} iterations with barycentre error {err_barycentres} and potential error {err_potentials}')
    
    if debiasing:
        # Attach potential to the graph - all pointing to the same item
        for edges in dp.graph.edges:
            dp.data_dict[edges[0]]['debiased_potential'] = d
        
        for edges in dp.graph.edges:
            assert dp.data_dict[edges[0]]['debiased_potential'] is d, "Debiasing potential should be the same object"

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
    for e1, e2, w in dp.graph.edges(data=True):
        s = _chizat_reduction_for_sinkhorn(dp, e2, (e1, e2), epsilon)
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
        if 'x1y1' in dp.data_dict[edge2] and 'x2y2' in dp.data_dict[edge2]:
            # we can tensorise, so we must be able to tensorize the symmetric problem

            # Does eveyone have the same grid?
            assert edge1 != edge2
            if dp.data_dict[edge1]['x1y1'] is dp.data_dict[edge2]['x1y1']:
                # then the edges are sharing a grid
                # So assign to the firs edge barycentre node the x1x1, x2x2
                dp.data_dict[edge2[0]]['x1x1'] = dp.data_dict[edge2]['x1y1']
                dp.data_dict[edge2[0]]['x2x2'] = dp.data_dict[edge2]['x2y2']
            else:
                # Not eveyone shares the same grid so need to compute symmetric version
                # it should still have a grid associated
                grid = dp.data_dict[edge2[0]]['grid']
                if isinstance(grid, tuple) :
                    dp.data_dict[edge2[0]]["x1x1"], dp.data_dict[edge2[0]]["x2x2"] = dp._cost_for_tuple(grid, grid)

                elif len(grid.shape) == 3 and len(grid.shape) == 3:
                    n1, n2, n3 = grid.shape
                    assert n3 == 2 , "We assume 2D points"

                    # Calculate cost matrices - the indexing works
                    # because torch cdist eliminats the common axis which will have the same values.
                    dp.data_dict[edge2[0]]["x1x1"], dp.data_dict[edge2[0]]["x2x2"] = dp._cost_for_meshgrid(grid, grid, n1, n2, n1, n2)
                    

        elif 'grid' in dp.data_dict[edge2[0]]:
            pass # we can use PyKeOps
    
def marginals(dp, epsilon, debiasing=True, nodes=None):
    """
    Calculate the marginals at the node(s) specified

    ToDo: Its not ideal that we have have to pass on debiasing... hummm
    """

    # Process nodes input:
    if nodes is None:
        nodes = list(dp.graph.nodes)
    elif isinstance(nodes, int):
        nodes = [nodes]
    elif isinstance(nodes, list):
        pass
    else:
        raise TypeError("Invalid nodes input")

    # Compute marginals
    marginals = {}
    for node in nodes:
        marginals[node] = {}
        marginals[node]['marginal'], marginals[node]['error'] = calculate_node_marginal(dp, node, epsilon, debiasing)

    return marginals

def calculate_node_marginal(dp, node, epsilon, debiasing):
    """
    Calculate the marginal for a specific node.
    """
    # Get the node's data
    node_data = dp.data_dict[node]

    # I have to look across my neighbours to see whos connected, and sum inwards;
    # I'm going to first do this only for the pairwise approach this its a lot more simple than 
    # the general graph case

    # Compute the marginal - depending on tensorisation or not
    for neighbour in dp.graph.neighbors(node):
        edge = (node, neighbour) if (node, neighbour) in dp.graph.edges else (neighbour, node)

        if debiasing:
            if hasattr(dp.data_dict[node], 'debiased_potential'):
                b = dp.data_dict[node]['a'] * dp.data_dict[node]['debiased_potential']
                a = dp.data_dict[neighbour]['a']
            elif hasattr(dp.data_dict[neighbour], 'debiased_potential'):
                b = dp.data_dict[node]['a']
                a = dp.data_dict[neighbour]['a'] * dp.data_dict[neighbour]['debiased_potential']
            else:
                raise Warning("No debiasing potentials attached to either node, yet using debiasing")
            
            if hasattr(dp.data_dict[node], 'debiased_potential') and hasattr(dp.data_dict[neighbour], 'debiased_potential'):
                raise Warning("Both nodes have debiasing potentials attached, this is unexpected behaviour")
        else:
            a = dp.data_dict[neighbour]['a']
            b = dp.data_dict[node]['a']  

        if 'x1y1' in dp.data_dict[edge] and 'x2y2' in dp.data_dict[edge]:
            # we can tensorise 
            marginal = _tensorised_marginal_reduction(
                dp.data_dict[edge]['x1y1'], # either order tensorise_f will sort it
                dp.data_dict[edge]['x2y2'],
                epsilon,
                a,
                b,
            )
        elif 'grid' in dp.data_dict[node] and 'grid' in dp.data_dict[neighbour]:
            # we can use PyKeOps
            marginal = _flat_grid_marginal_reduction(
                dp.data_dict[neighbour]['grid'],
                dp.data_dict[node]['grid'],
                epsilon,
                a,
                b,
            )

    error = torch.norm(
        marginal - node_data['density'],
        p=float('inf')
    ).item()

    return marginal, error


# If debiasing we can 'attach' the debiasing potential to the marginal reduction
# of a or b.
def _tensorised_marginal_reduction(x1y1, x2y2, epsilon, ai, bj):
    return tensorise_f(
        torch.exp(-x1y1/epsilon),
        torch.exp(-x2y2/epsilon),
        ai,
    ) * bj

def _flat_grid_marginal_reduction(X, Y, epsilon, ai, bj):
    return chizat_marginals(
        X,
        Y,
        epsilon,
        ai,
        bj,
    )


def asymmetric_cost(
        dp: SinkhornDataProcessor,
        epsilon,
        rho, 
        aprox: str,
        debiasing: bool = True,
        verbose: bool = False,
):  
    
    epsilon = dp._torch_numpy_process(epsilon)
    rho = dp._torch_numpy_process(rho)


    us_e = []
    for edge in dp.graph.edges:
        weighting = dp.graph.edges[edge]['weight']
        unbal_sinkhorn_div = _asymmetric_individual_cost()
        us_e.append(unbal_sinkhorn_div * weighting)

    if debiasing:
        # We need the last few terms
        d = dp.data_dict[edge[0]]['debiased_potential']
        debiasing_term = _calculate_debiasing_potential_symmetric_term(d, dp, edge[0], epsilon)

    return sum(us_e)- epsilon * debiasing_term/2,  us_e

def _asymmetric_individual_cost(
        dp,
        edge,
        epsilon,
        rho,
        aprox,
        debiasing
):
    bary_node= edge[0]
    data_node = edge[1]

    if debiasing:
        if hasattr(dp.data_dict[bary_node], 'debiased_potential'):
            b = dp.data_dict[bary_node]['a'] * dp.data_dict[bary_node]['debiased_potential']
            a = dp.data_dict[data_node]['a']
        elif hasattr(dp.data_dict[data_node], 'debiased_potential'):
            raise Warning("No debiasing potentials should be attached to the data")
        else:
            raise Warning("No debiasing potentials attached to either node, yet using debiasing")

        if hasattr(dp.data_dict[bary_node], 'debiased_potential') and hasattr(dp.data_dict[data_node], 'debiased_potential'):
            raise Warning("Both nodes have debiasing potentials attached, this is unexpected behaviour")
    else:
        a = dp.data_dict[data_node]['a']
        b = dp.data_dict[bary_node]['a']  
    
    # Have sufficent information for term 1 and term 2 of dual cost
    term1 = _dual_cost_data_term(a, dp.data_dict[data_node]['density'], aprox, epsilon, rho)
    term2 = _dual_cost_data_term(a, dp.data_dict[data_node]['density'], 'balanced', epsilon, rho)
    term3 = calculate_node_marginal(dp, bary_node, epsilon, debiasing)[0].sum()

    # final constant <K>
    term4 = _calculate_dual_cost_constant(dp, edge, epsilon, debiasing)

    return term1 + term2 - epsilon*(term3 - term4)
    
def _calculate_dual_cost_constant(dp, edge, epsilon, debiasing):
    """
    we can hack the marginal reductions for find the cost constant summation <K>
    by using ones vectors for ai and bj
    """

    bary_node= edge[0]
    data_node = edge[1]

    if debiasing:
        if hasattr(dp.data_dict[bary_node], 'debiased_potential'):
            b = torch.ones_like(dp.data_dict[bary_node]['a']) * dp.data_dict[bary_node]['debiased_potential']
            a = torch.ones_like(dp.data_dict[data_node]['a'])
        elif hasattr(dp.data_dict[data_node], 'debiased_potential'):
            raise Warning("No debiasing potentials should be attached to the data")
        else:
            raise Warning("No debiasing potentials attached to either node, yet using debiasing")

        if hasattr(dp.data_dict[bary_node], 'debiased_potential') and hasattr(dp.data_dict[data_node], 'debiased_potential'):
            raise Warning("Both nodes have debiasing potentials attached, this is unexpected behaviour")
    else:
        a = torch.ones_like(dp.data_dict[data_node]['a'])
        b = torch.ones_like(dp.data_dict[bary_node]['a'])

    if 'x1y1' in dp.data_dict[edge] and 'x2y2' in dp.data_dict[edge]:
        # we can tensorise 
        cost_constant = _tensorised_marginal_reduction(
            dp.data_dict[edge]['x1y1'], # either order tensorise_f will sort it
            dp.data_dict[edge]['x2y2'],
            epsilon,
            a,
            b,
        )
    elif 'grid' in dp.data_dict[data_node] and 'grid' in dp.data_dict[bary_node]:
        # we can use PyKeOps
        cost_constant = _flat_grid_marginal_reduction(
            dp.data_dict[data_node]['grid'],
            dp.data_dict[bary_node]['grid'],
            epsilon,
            a,
            b,
        )

    return cost_constant.sum()

def _calculate_debiasing_potential_symmetric_term(d, dp, node, epsilon):
    """
    we can hack the marginal reductions for find the cost constant summation <K>
    by using ones vectors for ai and bj
    """

    if 'x1x1' in dp.data_dict[node] and 'x2x2' in dp.data_dict[node]:
        # we can tensorise 
        cost_constant = _tensorised_marginal_reduction(
            dp.data_dict[node]['x1x1'], # either order tensorise_f will sort it
            dp.data_dict[node]['x2x2'],
            epsilon,
            d-1,
            d-1,
        )
    elif 'grid' in dp.data_dict[node]:
        # we can use PyKeOps
        cost_constant = _flat_grid_marginal_reduction(
            dp.data_dict[node]['grid'],
            dp.data_dict[node]['grid'],
            epsilon,
            d-1,
            d-1,
        )

    return cost_constant.sum()
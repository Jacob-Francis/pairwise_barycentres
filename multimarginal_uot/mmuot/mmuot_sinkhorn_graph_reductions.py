import numpy
from .pykeops_formulaes import alpha_reduction_pykeops
import torch
from graph_dp import SinkhornDataProcessor
from .utils import aprox_lse_update, alpha_reduction
import networkx as nx
import numpy as np

def graph_creator_from_edges_and_weights(edges, weights=None):
    import networkx as nx

    if weights is None:
        weights = [1/len(edges) for _ in edges]

    G = nx.Graph()
    for i, edge in enumerate(edges):
        G.add_edge(edge[0], edge[1], weight=weights[i])
    return G


def generate_mmuotdataprocessor_star_graph(data, grid=None, weights=None, cuda_device=None, clear_grid=True):
    """
    # Data should be arranged as a list of lists
    # i.e. data[i] = [density, grid]
    # the grid and density could be [None, None]
    # data[0] is the central node - this can be None (i.e. uniform density on barycentre grid)

    grid if sharing all the same grid
    
    """
    # Build graph
    M = len(data)
    edges = []
    counter = 0
    data_dict = {}
    for i in range(M):
        # Building star all from node 0
        data_dict[counter] = {
            'density': data[i][0], # this is the bayrcentre which will have a uniform density to start
            'grid': grid if grid is not None else data[i][1],
        }
        counter += 1
    
    # Star graph edges
    for i in range(1, M):
        edges.append((0, i))

    graph = graph_creator_from_edges_and_weights(edges, weights)

    # build data processor
    dp = SinkhornDataProcessor(
        potentials='f',
        data_dict=data_dict,
        graph=graph,
        free_grids=False,
        grid=grid,
        cuda_device=cuda_device,
    )

    # Add alpha - adding edge keys if necessary
    for edge in dp.graph.edges:
        dp.data_dict.setdefault(edge, {})['alpha'] = torch.ones_like(dp.data_dict[edge[0]]['f'])
        dp.data_dict.setdefault((edge[1], edge[0]), {})['alpha'] = torch.ones_like(dp.data_dict[edge[1]]['f'])

    # If tenosrising then we need to add the tensorisation grids - since tensori_f decides on
    # oreintation we can just point to the same grids

    for edge in dp.graph.edges:
        if 'x1y1' in dp.data_dict[edge]:
            dp.data_dict[(edge[1], edge[0])]['x1y1'] = dp.data_dict[edge]['x1y1']
            dp.data_dict[(edge[1], edge[0])]['x2y2'] = dp.data_dict[edge]['x2y2']

    # clean memory
    # We can't clean all memory we can only clean the grids that are not used by the data
    # we can also make sure the barycentre grid is pointing to the same grid.
    if clear_grid:
        for edge in dp.graph.edges:
            if 'x1y1' in dp.data_dict[edge]:
                try:
                    del dp.data_dict[edge[0]]['grid']
                    del dp.data_dict[edge[1]]['grid']
                except (KeyError):
                    pass

    return dp

def generate_mmuotdataprocessor(data, graph, grid=None, weights=None, cuda_device=None):
    """
    # Data should be arranged as a list of lists
    # i.e. data[i] = [density, grid]
    # the grid and density could be [None, None]
    # data[0] is the central node
    """
    raise NotImplementedError("General graph generation not implemented yet")
   


def procesinng_data_holder_for_mmot():
    pass
# process the dictionary to have an alpha variable on each edge (and in both directions)
# but for tensorisation we don't need to store in both directions
# As part of the processing make sure that x1y1 and x2y2 are avaliable from both orderes of the edge .


def sinkhorn_update(dp: SinkhornDataProcessor, j, epsilon, rho, aprox='balanced', prod=True):
    """
    Usual sinkhorn reduction though now over many branches using the recursive reduction done before
    """
    
    temp = torch.ones_like(dp.data_dict[j]['f'])

    # ToDo i think i shoudl be able to stablaise this since we have the max weighted shift inside 
    # pykeops, and then we take log
    for i in dp.graph.neighbors(j):
        temp *= dp.data_dict[(j, i)]["alpha"]

    # Log-sum-exp
    if prod:
        temp = - epsilon * torch.log(temp)
    else:
        temp = epsilon * torch.log(dp.data_dict[j]["density"]) - epsilon*torch.log(temp)

    # pointwise aprox;
    temp = aprox_lse_update(temp, epsilon, rho, aprox=aprox)

    # psuedo convergence err
    err = torch.linalg.norm(
        temp.view(-1, 1) - dp.data_dict[j]["f"].view(-1, 1),
        ord=float("inf"),
    ).item() # noqa: E1102

    return temp, err

def mmuot_sinkhorn_loop(dp: SinkhornDataProcessor,
    epsilon,
    rho,
    max_iterations=20,
    tol=1e-7,
    aprox='balanced',
    prod=True,
    convergence_tracking=False,
    verbose=False,
    ):

    # Initialisations
    err = tol + 1.
    count = 0

    if convergence_tracking:
        convergence_tracking = []

    # root node
    v0 = list(dp.graph.nodes)[0]

    # Depth first traversal to do reductions
    dfs_edges = list(nx.dfs_tree(dp.graph, source=v0).edges)
    
    # First initialisation of alphas'
    # ToDo: do i save internally inside alpha reduction?
    for p_j, j in reversed(dfs_edges):
        dp.data_dict[(p_j, j)]['alpha'] = alpha_reduction(dp, p_j, j, epsilon, prod=prod)
    
    # Begin Sinkhorn loop:
    while count < max_iterations and err > tol:
        err = -torch.inf
        
        # root node isn't enforces when going over edges
        # Sinkhorn update (including aprox relaxation)
        dp.data_dict[v0]['f'], er = sinkhorn_update(dp, v0, epsilon, rho, aprox=aprox, prod=prod)
        err = max(err, er)

        for p_j, j in dfs_edges:
            # update alpha
            if j != v0:
                dp.data_dict[(j, p_j)]['alpha'] = alpha_reduction(dp, j, p_j, epsilon, prod=prod)
            
            # Sinkhorn update (including aprox relaxation)
            dp.data_dict[j]['f'], er = sinkhorn_update(dp, j, epsilon, rho, aprox=aprox, prod=prod)

            err = max(err, er)
        
        # update reverse alpha
        for p_j, j in reversed(dfs_edges):
            dp.data_dict[(p_j, j)]['alpha'] = alpha_reduction(dp, p_j, j, epsilon, prod=prod)
        
        count += 1

        if convergence_tracking:
            convergence_tracking.append(err.item())
        
        if verbose:
            _, e = mmuot_marginal_j(dp, j, epsilon, prod=prod, update_alpha=False)
            print(f"Iteration {count}, Error: {err}, Mar' Err: {e}")

    return dp

def mmuot_marginal_j(dp: SinkhornDataProcessor, j, epsilon, prod=True, update_alpha=False):
    
    marginal = torch.ones_like(dp.data_dict[j]['f'])

    # Gather incoming alphas
    for i in dp.graph.neighbors(j):
        if update_alpha:
            alpha_reduction(dp, j, i, epsilon, prod=prod)
        marginal *= dp.data_dict[(j, i)]['alpha']
    
    if prod:
        marginal = torch.exp(dp.data_dict[j]['f']/epsilon) * marginal * dp.data_dict[j]['density']
        print('sums', marginal.sum(), dp.data_dict[j]['density'].sum())
    else:
        marginal = torch.exp((dp.data_dict[j]['f'])/epsilon) * marginal / np.prod(dp.data_dict[j]['f'].shape)
    
    # print('SUMs',/ marginal.sum(), dp.data_dict[j]['density'].sum())
    # marginal errror
    err = torch.linalg.norm(
        marginal.view(-1) - dp.data_dict[j]['density'].view(-1), ord=float("inf")
    ).item()

    return marginal, err

def mmuot_marginals(dp: SinkhornDataProcessor, epsilon, prod=True, alpha_update=False):
    
    marginals = {}
    errors = {}

    for j in dp.graph.nodes:
        marginals[j], errors[j] = mmuot_marginal_j(dp, j, epsilon, prod=prod, update_alpha=alpha_update)
    
    return marginals, errors


def mmuot_dual_cost(dp: SinkhornDataProcessor, epsilon, rho, aprox='balanced', prod=True, debiasing=False):
    
    cost = 0.0

    return cost